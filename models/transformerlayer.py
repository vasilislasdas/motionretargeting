import time
import torch
import torch.nn as nn
from multiheadattention import MultiHeadAttention
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# class that implements the feedforward layer in the transformer
class ForwardLayer( nn.Module ):

    def __init__(self, dim_input: int , dim_inter: int  = None):
        super( ForwardLayer, self).__init__()

        """
        Description: 
            Implements the forward layer in a transformer: Linear layer -> Relu-> Linear layer
        Args:
            dim_input( int ): dimensionality of input  
            dim_intermediate( int ): dimensionality of the output of the first layer           
        Returns:
            New representation of the input(the output of the attention module)        
        Note: 
            The first layer maps the input dim to another dim(dim_intermediate), then relu is applied, and the final layer 
            reprojects back to the same input dim. As a whole the operation of this module leaves the dimensions 
            of the input untouched.
        Note: 
            If the intermediate positions is unspecified then, it is the same input dim. E.g. if dim_input = 512,
            then dim_inter = 512. No need for fancy special intermediate dimensions         
        """

        self.dim_input = dim_input
        self.dim_inter = dim_input if dim_inter is None else dim_inter

        # layers:2 linear layers and one relu in between
        self.layer1 = nn.Linear( self.dim_input, self.dim_inter )
        self.relu = nn.ReLU()
        # self.relu = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear( self.dim_inter, self.dim_input )


    def forward( self, input ):

        # test to make sure we operate over batches: batch_size X seq_len x dim_input
        assert( input.ndim == 3 )

        # create directly the output
        out = self.layer2( self.relu( self.layer1( input ) ) )

        return out





# implements a transformer-encoder layer
class EncoderLayer( nn.Module ):

    def __init__( self, dim_model: int, nr_heads: int = 1, dropout : float = 0.1, dim_inter: int = None ):
        super( EncoderLayer, self).__init__()

        """
        Description: 
            Implements a classical encoder transformer layer
        Args:
            dim_model: dimensionality of input and also hidden dim              
            nr_heads( int ): number of heads in multi-head attention
            dropout(float): dropout rate 
            dim_intermediate( int ): dimensionality of the output of the first layer of the FFN
                       
        Returns:
            New representation of the input after attention+ffn and residual connections+layernormalizaion of the output 
            of attention and ffn  
        Note:
        Note: the encoder layer uses only one dimension: the dimension of the input which is the same as the internal 
               dimension of the representations
            Steps: 1) input -> layernorm( input + dropout( attention( input) ) ) -> attention_output
                    i) calcaulte MHA
                    ii) use dropout
                    iii) layer-normalize the input  + dropout of attention output
                   2) layernorm( attention_output + dropout( ffd(attention_output)  ))
                    i) output of the previous sublayer gets to a ffn net -> ffn_output
                    ii) dropout is applied to the ffn_output
                    iii) layernormalize : attention_output + dropout(ffn_output)                  
        Note:
            For stacked transformer-encoder layers, the input-output dims are the same, with the exception of the first
            layer which gets fed as input the embeddings(NLP) or something else that might have different dimensionality      
        
        Note:
            The input dim of the ffn layer is the same as the output dimension of the MHA module: dim_hidden            
        """

        # store internally parameters and use onle these from now on
        self.dim_model = dim_model
        self.nr_heads = nr_heads
        self.dropout = dropout
        self.dim_inter = dim_inter

        ### create the modules that build up the encoder layer ##
        # create a multihead attention module
        self.mha = MultiHeadAttention( dim_input= self.dim_model, dim_hidden=self.dim_model, nr_heads=self.nr_heads )

        # create a ff layer: notice that the input dim to the layer is the dim_model as everywhere
        self.ffn = ForwardLayer( dim_input = self.dim_model, dim_inter=self.dim_inter )

        # create 2 layernormalized + dropoutlayers
        self.layernorm1 = nn.LayerNorm( self.dim_model )
        self.layernorm2 = nn.LayerNorm( self.dim_model )
        self.drop1 = nn.Dropout( p=self.dropout )
        self.drop2 = nn.Dropout(p=self.dropout)

        # that's all folks!

    def forward(self, input ):

        assert( input.ndim == 3 )

        # get new represeantions from mha + attention scores for each head
        reps, attention = self.mha(input_queries=input,input_keys=input,input_values=input)

        # dropout MHA output
        reps = self.drop1( reps )

        # layernormalize to get the final MHA output
        mha_out = self.layernorm1( input + reps )

        # feed input to the FFN laayer
        ffn_out = self.ffn( mha_out )

        # dos same: dropout+layernormazlie
        ffn_out = self.drop2( ffn_out )
        final_out = self.layernorm2( mha_out + ffn_out )

        # finally retrun the total output + attention scores for visualization
        return final_out, attention


# implements a transformer-decoder layer
class DecoderLayer(nn.Module):

    def __init__( self, dim_model: int, nr_heads: int = 1, dropout: float = 0.1, dim_inter: int = None):
        super(DecoderLayer, self).__init__()

        """
        Description: 
            Implements a vanilla decoder transformer layer
        Args:
            dim_model( int ): dimension from the input to the decoder: target sequence and encoder_reps               
            nr_heads( int ): number of heads in multi-head attention(same for self and cross attention)
            dropout(float): dropout rate 
            dim_intermediate( int ): dimensionality of the output of the first layer of the FFN

        Returns:
            New representation of the input after attention+ffn and residual connections+layernormalizaion of the output 
            of attention and ffn + using the output representation from the encoder for cross-attention 
        Note:
            The decoder layer is almost identical to the encoder layer, with the exception that it takes also as input the 
            encoder representations for cross-attention
        Note:
            The input in the decoder layer is the target in that case. It needs also the encoder reps or any reps.
        Note:
            Steps: 1) self-attention, dropout, residual + layernorm
                   2) cross-attention, dropout, residual + layernorm
                   3) feedforward net as in the  encoder                  
        Note:
            For stacked transformer-encoder layers, the input-output dims are the same, so we use one dimension: dim_model            
        Note:
            The input dim of the ffn layer is the same as the output dimension of the MHA module: dim_model        
        """

        # store internally parameters and use onle these from now on
        self.dim_model = dim_model
        self.nr_heads = nr_heads
        self.dropout = dropout
        self.dim_inter = dim_inter

        ### create the modules that build up the Decoder layer ##
        # masked-multihead attention module for the input
        self.masked_mha = MultiHeadAttention( dim_input=self.dim_model, dim_hidden=self.dim_model, nr_heads=self.nr_heads)

        # cross-attention
        self.cross_mha = MultiHeadAttention( dim_input=self.dim_model, dim_hidden=self.dim_model, nr_heads=self.nr_heads )

        # create a ff layer: notice that the input dim to the layer is the dim_hidden, which is output of the MHA
        self.ffn = ForwardLayer( dim_input=self.dim_model, dim_inter=self.dim_inter )

        # create 3 layernormalized + dropoutlayers
        self.layernorm1 = nn.LayerNorm(self.dim_model)
        self.layernorm2 = nn.LayerNorm(self.dim_model)
        self.layernorm3 = nn.LayerNorm(self.dim_model)
        self.drop1 = nn.Dropout(p=self.dropout)
        self.drop2 = nn.Dropout(p=self.dropout)
        self.drop3 = nn.Dropout(p=self.dropout)

        # that's all folks!


    # target is the input to the decoder: we need to pass the encoder reps before
    def forward( self, target:torch.Tensor, encoder_reps:torch.Tensor  ):
        assert( target.ndim == 3 )
        assert( encoder_reps.ndim == 3 )
        assert( target.size()[-1] == encoder_reps.size()[-1] )

        ### self-masked attention
        self_reps, self_attention = self.masked_mha( input_queries=target,input_keys=target, input_values=target,mask=True )

        # dropout self_masked attention
        self_reps = self.drop1( self_reps )

        # layernormalize to get the output of the self-attention in the decoder
        self_mha_out = self.layernorm1( target + self_reps )

        ### cross-attention
        cross_reps, cross_attention = self.cross_mha( input_queries=self_mha_out, input_keys=encoder_reps, input_values=encoder_reps)

        # dropout self_masked attention
        cross_reps = self.drop2( cross_reps)

        # layernormalize
        cross_mha_out = self.layernorm2( cross_reps + self_mha_out )

        # feed input to the FFN laayer
        ffn_out = self.ffn( cross_mha_out )

        # dos same: dropout+layernormazlie
        ffn_out = self.drop3( ffn_out )
        final_out = self.layernorm3( cross_mha_out + ffn_out )

        # finally return the total output + all attention scores for visualization
        return final_out, self_attention, cross_attention







