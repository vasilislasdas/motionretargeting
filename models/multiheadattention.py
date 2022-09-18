

import torch
import torch.nn as nn
import torch.nn.functional as Func
from PIL.ImageWin import HDC
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
import time
import numpy as np


# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, nr_heads: int ):
        super(MultiHeadAttention,self).__init__()

        """
        Description: 
            Computes multi-headed attention of the input
        Args:
            dim_input( int ): dimensionality of input  
            dim_hidden( int ): dimensionality of total hidden representations
            nr_heads( int ): heads used for attention              
        Returns:
            New representation of the input and the attention scores: torch.Tensor, torch.Tensor        
        Note: 
            We support only common dimensions for queries keys and values.            
        Note: 
            The hidden dimension is for all the heads. In practice each head uses dim_hidden/nr_heads
        Note:
            The input is of the form: batch_nr x seq_len x dim_input. The output is batch_nr x seq_len x dim_hidden.           
            The attention dimensions are: batch_nr x seq_len x  seq_len.           
        Note:
            The input_dim is split into nr_heads and in each head(subdimension), attention is applied. E.g. if dim_hidden = 512
            and nr_heads 8, first the input is projected only once to query, keys and values globally(speeding up things). 
            Then, each head is responsible for 512 / 8 = 64 dimensions. The new_representations at each head(e.g. 64 dimension)
            are concatenated and passed through a final layer.  
        Note: supports both masked and normal attention: pass the mask value in the forward function
        """

        # TODO: check how the heads are combined properly

        # store internally the parameters
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.nr_heads = nr_heads

        # test to make sure that the hidden dimensionality must be a multiple of heads
        assert( self.dim_hidden % self.nr_heads == 0  )

        # store internally the dimension of each head
        self.dim_head = self.dim_hidden / self.nr_heads

        # create the layers that generate the queries keys and values
        self.Wq = nn.Linear( self.dim_input, self.dim_hidden )
        self.Wk = nn.Linear( self.dim_input, self.dim_hidden )
        self.Wv = nn.Linear( self.dim_input, self.dim_hidden )

        # create final linear layer that just does an extra linear projection of the concatenated
        # representations of each head. Notice the input-output dimensions are the same
        self.Wo = nn.Linear( self.dim_hidden, dim_hidden )

    # masked or not, multihead attention
    def fast_attention( self, queries:torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask=False ):

        # split the projections to nr_heads: batch_nr X seq_len X dim_hidden -> batch_nr X nr_heads X seq_len X dim_hidden/nr_heads
        new_queries = rearrange( queries, " batch seq_len ( heads dim_head )-> batch heads seq_len dim_head", heads=self.nr_heads )
        new_keys = rearrange(keys, " batch seq_len ( heads dim_head )-> batch heads seq_len dim_head", heads=self.nr_heads)
        new_values = rearrange(values, " batch seq_len ( heads dim_head )-> batch heads seq_len dim_head", heads=self.nr_heads)

        # print(f"Diagnostic of multiple head:{new_queries.shape}, {new_keys.shape}, {new_values.shape}")

        # dividor for scaled-dot product
        dividor = self.dim_head ** 0.5

        # compute scaled dot product for each head direcly
        scores_einsum = torch.einsum("mnik,mnjk->mnij", [new_queries, new_keys])
        attention = torch.div( scores_einsum, dividor )

        # mask attention scores: needed for the decoder
        if mask == True :
            # print("Masking attention scores")
            mask = torch.tril( torch.ones_like( attention ) )
            attention = attention.masked_fill( mask == 0, -1e9 )

        # softmax the scores
        attention = torch.nn.functional.softmax( attention, dim=-1)

        # compute new representations
        new_reps = torch.einsum( "mnik,mnkj->mnij", [attention, new_values])

        # concatenate the new represenations from each head: batch_nr X nr_heads X seq_len X dim_head -> batch_nr X seq_len X nr_heads*dim_head
        final_reps = rearrange(new_reps, " batch heads seq_len dim_head-> batch seq_len ( heads dim_head )", heads=self.nr_heads)

        return final_reps, attention


    def forward(self, input_queries:torch.Tensor, input_keys: torch.Tensor, input_values: torch.Tensor, mask: bool = False  ):

        # we expect input to have 3 dims: batch_nr X seq_len X input_dim
        assert( input_queries.ndim == 3 )
        assert (input_keys.ndim == 3)
        assert (input_values.ndim == 3)

        # print some diagnostic info
        # print( f"Input Dimensions:{ input.shape }")

        # project input to query keys values: can be the same for self-attention, but different for cross attention
        queries = self.Wq(input_queries)
        keys = self.Wk(input_keys)
        values = self.Wv(input_values)

        # get new represeantions of the input + the attention scores
        reps, attentions = self.fast_attention( queries=queries, keys=keys, values=values, mask=mask )

        # pass the new representations through a final linear layer
        reps = self.Wo( reps )

        return reps,attentions


# simple query,key, based attention: no matrix form
class AttentionHeadSingle(nn.Module):

    def __init__(self, dim_input: int , dim_qkv:int ):
        super(AttentionHeadSingle, self).__init__()

        # query key values share the same hidden dimension
        # store internally the dimensions of input and output
        self.dim_input = dim_input
        self.dim_qkv = dim_qkv

        # create linear layer to project the input to queries, keys, values
        self.Wq = nn.Linear( dim_input, dim_qkv )
        self.Wk = nn.Linear( dim_input, dim_qkv )
        self.Wv = nn.Linear( dim_input, dim_qkv )

    # def attention( query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor )-> (float, torch.Tensor):
    def slow_attention( self, query:torch.Tensor,keys: torch.Tensor, values:torch.Tensor):
        # takes as input a 1xD vector
        assert( query.ndim == 1 )
        assert (keys.ndim == 2)
        assert (values.ndim == 2)

        # dot product + normalization
        dividor = self.dim_qkv ** 0.5
        scores = torch.div(  torch.mv( keys, query ), dividor  )
        scores = Func.softmax( scores, dim=0 )

        # compute new representations
        rep = torch.mv(values.t(), scores)
        return rep

    def fast_attention( self, queries:torch.Tensor,keys: torch.Tensor, values:torch.Tensor, mask = False ):

        # queries, keys, values: batch x seq_length X h_dim each :operate direcly on all queries keys and values
        # e.g. 1x6x3: 1 == batch_size, 6 == seq_length, 3 = hidden_dim

        # some tests
        assert( queries.ndim == 3 )
        assert (keys.ndim == 3 )
        assert (values.ndim == 3)

        # calculate scaled dot-product
        dividor = self.dim_qkv ** 0.5
        attention =  torch.div( torch.bmm( queries, torch.permute( keys, (0,2,1) ) ), dividor )

        # mask attention scores: needed for the decoder
        if mask:
            print("Masking attention scores")
            mask = torch.tril( torch.ones_like(attention) )
            attention = attention.masked_fill( mask == 0, -1e9 )

        # normalize: notice that we normalize over dim=2:
        attention = Func.softmax( attention, dim=2 )

        # calculate new represenations
        rep = torch.bmm( attention, values )
        return rep, attention



    def forward(self, input: torch.Tensor):

        batch_size, seq_len, input_dim = input.size()
        print( "Input size:", input.size(), " input dims:", input.ndim)

        # calculate queries key values directly in batch mode
        queries = self.Wq(input)
        keys = self.Wk(input)
        values = self.Wv(input)

        # the new representations
        new_represenations = torch.zeros( ( batch_size, seq_len, self.dim_qkv ) )

        print( f"New Reprs size:{new_represenations.size()}")

        # stupid slow for-loop to fetch each query key value and get attention
        for i in range( batch_size ):
            for j in range( seq_len ):
                query =queries[0, j ]
                tmp = self.slow_attention( query=query, keys=keys[i], values=values[i] )
                new_represenations[ i,j,: ] = tmp

        # fast represenations
        fast_reps,attention = self.fast_attention( queries=queries, keys=keys, values=values )

        # print(new_represenations[0,3]),  print(fast_reps[0,3])

        return fast_reps





if __name__ == '__main__':

    batch, dim_input, dim_output, seq_len, nr_heads = 1, 512, 512, 60, 8
    inp = torch.rand( batch, seq_len, dim_input )
    print( f"Input:{inp.size()}")

    # inp = inp.to(device="cuda")
    # multihead attention

    net = MultiHeadAttention(dim_input=dim_input,dim_hidden=dim_output,nr_heads=nr_heads )
    # net = net.to(device="cuda")
    start = time.time()
    out, attention = net(inp,inp,inp,mask=True)
    print(f"Elapsed time:{time.time() - start}")
    # plt.matshow(attention.squeeze(0).detach().numpy()), plt.show()
    print("output:", out.size())
    print("attention:", attention.size())






