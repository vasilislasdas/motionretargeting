import einops
import torch.utils.data as data
import torch
import torch.nn as nn
import time
from preprocess import *
from train import *
from fid import calculate_frechet_distance
import numpy as np
def frechet_distance( real, fake):
    pass

ee_cat1 = torch.LongTensor([0, 4, 9, 16, 22, 27]) # 28 joints +root node: 0
ee_cat2 = torch.LongTensor([0, 4, 8, 14, 18, 22])  # 23 joints +root node: 0
ee_dist_cat1 = torch.LongTensor([ 4, 4, 6, 9, 9 ]) # distance between root and end effectors
ee_dist_cat2 = torch.Tensor( [ 4, 4, 6, 8, 8 ] ) # distance between root and end effectors
ee_dist_cat11 = ee_dist_cat1.unsqueeze(dim=1).repeat(1,3).to('cuda')
ee_dist_cat22 = ee_dist_cat2.unsqueeze(dim=1).repeat(1,3).to('cuda')




def compute_validation_loss_tmp(  validator, input_seq, input_skeleton):

    _, real_reps, _,_ = validator( input_seq, input_skeleton, input_skeleton )
    a,b,c = real_reps.shape
    real_reps = real_reps.reshape(a, b * c).cpu().detach().numpy()
    mu = np.mean(real_reps, axis=0)
    sigma = np.cov( real_reps, rowvar=False)
    np.savez("mu_sigma", mu_real=mu, sigma_real=sigma)
    # r = 1


def compute_validation_loss( retargeter, validator, input_seq, input_skeleton, mu, sigma ):

    # print("COMPUTING CYCLE LOSS!!")
    input_seq = input_seq.detach().clone()
    input_skeleton = input_skeleton.detach().clone()

    # pick up random target skeletons
    new_indices = torch.randperm(input_seq.shape[0])
    target_skeleton = input_skeleton[new_indices].detach().clone()

    # retarget
    retargeting, _, _,_ = retargeter( input_seq, input_skeleton, target_skeleton )
    assert (torch.any(torch.isnan(retargeting)) == False)
    assert (torch.any(torch.isinf(retargeting)) == False)

    # pass the real and generated data through the validator to get their representations
    _, fake_reps,_,_ = validator( retargeting, target_skeleton, target_skeleton )

    # compute statistic of real and generated data
    a,b,c = fake_reps.shape
    fake_reps = fake_reps.reshape(a, b*c).cpu().detach().numpy()
    mu2 = np.mean( fake_reps, axis=0 )
    sigma2 = np.cov( fake_reps, rowvar=False)

    fid = calculate_frechet_distance(mu1=mu, sigma1=sigma, mu2=mu2, sigma2=sigma2 )

    return fid






def get_self_module_losses_generator():

    # all self losses: rotations, positions for root node + latent space
    # + time difference losses

    # all loss modules
    self_loss_modules = []

    # losses for rotations + positons + root_node position
    self_loss_rotations = nn.MSELoss()
    self_loss_positions = nn.MSELoss()

    # time losses for all the above
    self_loss_time_rotations = nn.MSELoss()
    self_loss_time_positions = nn.MSELoss()

    # append all losses: 6 losses in total
    self_loss_modules.append(self_loss_rotations)
    self_loss_modules.append(self_loss_positions)
    self_loss_modules.append(self_loss_time_rotations)
    self_loss_modules.append(self_loss_time_positions)

    return self_loss_modules


def get_cycle_module_losses_generator():
    # all cycle losses: rotations, positions-root node, latent space
    # + time difference losses

    # all loss modules
    cycle_loss_modules = []

    # losses for rotations + root_node position + latent space
    cycle_loss_rotations = nn.MSELoss()
    cycle_loss_positions = nn.MSELoss()
    cycle_loss_latent = nn.MSELoss()

    # time losses for all the above
    cycle_loss_time_rotations = nn.MSELoss()
    cycle_loss_time_positions = nn.MSELoss()
    cycle_loss_time_latent = nn.MSELoss()

    # end effector loss also
    cycle_loss_ee = nn.MSELoss()

    # append all losses: 6 losses in total
    cycle_loss_modules.append( cycle_loss_rotations)
    cycle_loss_modules.append( cycle_loss_positions)
    cycle_loss_modules.append( cycle_loss_latent)
    cycle_loss_modules.append( cycle_loss_time_rotations)
    cycle_loss_modules.append( cycle_loss_time_positions)
    cycle_loss_modules.append( cycle_loss_time_latent)
    cycle_loss_modules.append(cycle_loss_ee)

    return  cycle_loss_modules


def compute_time_loss( time_loss_module, expected_seq, prediction ):

    # print(f"Input, output:{prediction.shape,expected_seq.shape}")

    # compute time-consistency loss
    predicted_diff = prediction[:, 1:, :] - prediction[:, :-1, :]
    expected_diff = expected_seq[:, 1:, :] - expected_seq[:, :-1, :]

    # time loss
    time_loss = time_loss_module( predicted_diff, expected_diff )

    return time_loss



def get_disc_loss( generator, discriminator,  input_seq, input_skeleton ):

    # random target skeletons + clone all
    input_seq = input_seq.detach().clone()
    input_skeleton = input_skeleton.detach().clone()
    new_indices = torch.randperm(input_seq.shape[0])
    target_skeleton = input_skeleton[new_indices].detach().clone()

    # compute retargeted data + classify them in the discrimininator
    prediction, _,_,_ = generator( input_seq, input_skeleton, target_skeleton )

    # detach the generated data!!
    prediction = prediction.detach() #

    # classify retargeted motions using the discriminator
    real, real_diff = discriminator( input_seq, input_skeleton )
    fake, fake_diff = discriminator( prediction, target_skeleton )

    # compute wasserstein loss for discriminator
    loss1 = -( torch.mean( real ) - torch.mean( fake ) )
    loss2 = -( torch.mean( real_diff) - torch.mean(fake_diff ) )
    loss = (loss1+loss2)/2

    return loss



def compute_gen_disc_loss( generator, discriminator,  input_seq, input_skeleton ):

    # random target skeletons + clone all
    input_seq = input_seq.detach().clone()
    input_skeleton = input_skeleton.detach().clone()
    new_indices = torch.randperm(input_seq.shape[0])
    target_skeleton = input_skeleton[new_indices].detach().clone()

    # compute retargeted data + get one single value for real-vs fake for each frame
    out, _,_,_ = generator( input_seq, input_skeleton, target_skeleton )
    fake, fake_diff = discriminator( out, target_skeleton )

    # use wasserstein loss
    loss1 = -torch.mean( fake )
    loss2 = -torch.mean( fake_diff )
    loss = (loss1+loss2)/2

    return loss



def denormalize_motion(motion, max, min ):

    delta = max - min
    denorm_motion = ((motion + 1) / 2) * delta + min
    isNan = torch.any( denorm_motion.isnan())
    isInf = torch.any(torch.isinf(denorm_motion))
    assert (isNan == False and isInf == False)

    return denorm_motion

def get_generator_loss( generator, discriminator, input_seq, input_skeleton,
                        self_loss_modules, cycle_loss_modules, epoch, extra_data, characters, h_orig ):


    all_losses = []

    ## compute self-reconstruction loss
    self_loss = compute_self_loss( model=generator, input_seq=input_seq, input_skeleton=input_skeleton, self_loss_modules=self_loss_modules, extra_data=extra_data )
    all_losses.extend(self_loss)

    # ## compute cycle-reconstruction loss:
    cycle_loss = compute_cycle_loss(model=generator, input_seq=input_seq, input_skeleton=input_skeleton,
                                    cycle_loss_modules=cycle_loss_modules, characters=characters, extra_data=extra_data, h_orig=h_orig)
    all_losses.extend(cycle_loss)

    # # compute discriminator loss every few epochs
    if (epoch % mod_epochs_gen) == 0 and epoch > 0:
        disc_loss = compute_gen_disc_loss(generator=generator, discriminator=discriminator,
                                          input_seq=input_seq, input_skeleton=input_skeleton )
        all_losses.append(disc_loss)

    return all_losses


def compute_self_loss( model, input_seq, input_skeleton, self_loss_modules, extra_data):


    # get edges and offsets
    edges = extra_data["edges"]
    offsets = extra_data["offsets"]
    max_motion = extra_data["max"].to( input_seq.device)
    min_motion = extra_data["min"].to(input_seq.device)

    input_seq = input_seq.detach().clone()
    input_skeleton = input_skeleton.detach().clone()

    # predict
    prediction, _, _,_ = model( input_seq, input_skeleton, input_skeleton )
    assert( torch.any( torch.isnan(prediction) )  == False )
    assert (torch.any( torch.isinf( prediction ) ) == False)


    # 1) fetch rotations from original and predicted and calculate rotation loss
    rotations_original = input_seq[:, :, 3:]
    rotations_predicted = prediction[ :, :, 3:]
    self_loss_rotations = self_loss_modules[0]( rotations_original, rotations_predicted )

    # 2 compute positions from original motion and predicted motion and calculate position_loss
    batch_size = int( input_seq.shape[0] / len( edges ) )
    # print(batch_size)

    # split motion back
    original_motion = torch.split( input_seq.clone(), batch_size )
    predicted_motion = torch.split( prediction.clone(), batch_size )
    assert( len(original_motion) == len(edges) )
    assert ( len(predicted_motion) == len(edges))

    pos_orig = []
    pos_pred = []
    # vel_orig = []
    # vel_pred = []

    for i in range(len(edges)):
        tmp_edge = edges[i]
        tmp_offset = offsets[ i ].to( input_seq.device )
        tmp_orig_mot = original_motion[ i ].to(input_seq.device)
        tmp_pred_mot = predicted_motion[ i ].to(input_seq.device)
        tmp_orig_mot = denormalize_motion(  motion=tmp_orig_mot, max=max_motion, min=min_motion )
        tmp_pred_mot = denormalize_motion(  motion=tmp_pred_mot, max=max_motion, min=min_motion )
        tmp_comb = torch.cat( [tmp_orig_mot, tmp_pred_mot], dim=0 )
        tmp_pos = calculate_positions_from_raw( tmp_comb, tmp_edge, tmp_offset )

        if len(tmp_edge) == 22:
            tmp_pos = tmp_pos[:,:,ee_cat2, : ]
        else:
            tmp_pos = tmp_pos[:, :, ee_cat1, :]

        tmp_pos = torch.chunk( tmp_pos, 2, dim = 0)
        tmp1 = tmp_pos[0]
        tmp2 = tmp_pos[1]
        pos_orig.append( tmp1 )
        pos_pred.append( tmp2 )


    pos_orig = torch.cat(pos_orig)
    pos_pred = torch.cat(pos_pred)
    pos_min = torch.amin( pos_orig, dim=(0,1))
    pos_max = torch.amax(pos_orig, dim=(0,1))
    # normalize the positions to 0-1
    pos_orig = (pos_orig - pos_min) / (pos_max - pos_min)
    pos_pred = (pos_pred - pos_min) / (pos_max - pos_min)
    pos_orig[torch.isnan(pos_orig)] = 0
    pos_orig[torch.isinf(pos_orig)] = 0
    pos_pred[torch.isnan(pos_pred)] = 0
    pos_pred[torch.isinf(pos_pred)] = 0



    # compute position loss for all positions, not only root
    self_loss_positions = self_loss_modules[1]( pos_orig, pos_pred )

    # 3) compute velocity(time_loss) for positions and rotations
    self_loss_rotations_time = compute_time_loss( time_loss_module=self_loss_modules[2], expected_seq=rotations_original, prediction=rotations_predicted )
    self_loss_positions_time = compute_time_loss(time_loss_module=self_loss_modules[3],  expected_seq=pos_orig, prediction=pos_pred)

    mult1 = 50
    mult2 = 20
    div1 = 2
    return [self_loss_rotations, self_loss_positions/div1,  mult1*self_loss_rotations_time, mult2*self_loss_positions_time ]






def compute_cycle_loss( model, input_seq, input_skeleton, cycle_loss_modules, characters, extra_data, h_orig):


    # get edges and offsets
    edges = extra_data["edges"]
    offsets = extra_data["offsets"]
    max_motion = extra_data["max"].to( input_seq.device)
    min_motion = extra_data["min"].to(input_seq.device)

    # get batch size
    batch_size = int( input_seq.shape[0] / len( edges ) )

    # clone input seq and skeleton just to make sure
    input_seq = input_seq.detach().clone()
    input_skeleton = input_skeleton.detach().clone()


    ##### pick up random target skeletons
    new_indices = torch.randperm(input_seq.shape[0])
    target_skeleton = input_skeleton[new_indices].detach().clone()

    ##### find which skeletons are in category1 and which in category2
    # get names of source and target chars
    unique_chars = []
    source_chars = []
    for i in range(len(characters)):
        source_chars += characters[i]
        unique_chars.append(characters[i][0])
    target_chars = [ source_chars[i] for i in new_indices.tolist() ]

    # scaling coeffs for end effector loss: make default for category 1

    # h_orig = einops.repeat(ee_distances, 'skel ee -> (skel batch) frames ee xyz', batch=batch_size, frames=29, xyz=3)
    h_target = h_orig[new_indices]

    # start_test = torch.randint_like(new_indices, low=0, high=100)
    start_test = torch.randperm(input_seq.shape[0])
    end_test = start_test[new_indices].clone()
    unique_indices = [ end_test.tolist().index(elem) for elem in start_test ]

    # tests
    # np.all(np.array(target_chars)[unique_indices] == np.array(source_chars))
    # np.all(np.array(target_chars)[unique_indices] == np.array(source_chars))

    #### cycle
    # retarget
    retargeting, motion_reps1, _,_ = model( input_seq, input_skeleton, target_skeleton )
    assert( torch.any( torch.isnan(retargeting) )  == False )
    assert (torch.any( torch.isinf( retargeting ) ) == False)

    # go back
    prediction, motion_reps2, _,_ = model(retargeting, target_skeleton, input_skeleton)
    assert (torch.any(torch.isnan(prediction)) == False)
    assert (torch.any(torch.isinf(prediction)) == False)


    # 1) fetch rotations from original and predicted and calculate rotation loss
    rotations_original = input_seq[:, :, 3:].clone()
    rotations_predicted = prediction[ :, :, 3:].clone()
    cycle_loss_rotations = cycle_loss_modules[0]( rotations_original, rotations_predicted )

    # 2) compute positions from original motion and predicted motion and calculate position_loss

    # split source motion and cycle-motion into batches
    original_motion = torch.split(input_seq.clone(), batch_size)
    predicted_motion = torch.split(prediction.clone(), batch_size)
    retargeted_motion =  retargeting[unique_indices].clone()
    retargeted_motion = torch.split( retargeted_motion, batch_size)
    assert (len(original_motion) == len(edges))
    assert (len(predicted_motion) == len(edges))
    assert (len(retargeted_motion) == len(edges))

    # original and target positions
    pos_orig = []
    pos_pred = []
    pos_ret = []

    # itearte over the edges: different skeletons
    for i in range(len(edges)):

        # get edge and offset
        tmp_edge = edges[i]
        tmp_offset = offsets[i].to(input_seq.device)

        # fetch motion from source and target skeleton for the batch
        tmp_orig_mot = original_motion[i].clone().to(input_seq.device)
        tmp_pred_mot = predicted_motion[i].clone().to(input_seq.device)
        tmp_ret_mot = retargeted_motion[i].clone().to(input_seq.device)

        # denormalize the motion to compute positions
        tmp_orig_mot = denormalize_motion(motion=tmp_orig_mot, max=max_motion, min=min_motion)
        tmp_pred_mot = denormalize_motion(motion=tmp_pred_mot, max=max_motion, min=min_motion)
        tmp_ret_mot = denormalize_motion( motion=tmp_ret_mot, max=max_motion, min=min_motion)

        # concatenate to avoid double position computation
        tmp_comb = torch.cat([tmp_orig_mot, tmp_pred_mot, tmp_ret_mot], dim=0)

        # compute positions for source and target
        tmp_pos = calculate_positions_from_raw( tmp_comb, tmp_edge, tmp_offset )

        # fetch positions of the end effectors based on the category
        if len(tmp_edge) == 22:
            tmp_pos = tmp_pos[:, :, ee_cat2, :]
        else:
            tmp_pos = tmp_pos[:, :, ee_cat1, :]

        # split positions back for source and target
        tmp_pos = torch.chunk(tmp_pos, 3, dim=0)
        tmp1 = tmp_pos[0].clone()
        tmp2 = tmp_pos[1].clone()
        tmp3 = tmp_pos[2].clone()

        # store positions for each skeleton-batch
        pos_orig.append(tmp1)
        pos_pred.append(tmp2)
        pos_ret.append( tmp3 )

        r = 1

    # concatenate in 3d all positions
    pos_orig = torch.cat(pos_orig)
    pos_pred = torch.cat(pos_pred)
    pos_ret = torch.cat(pos_ret)
    pos_orig_v2 = pos_orig.clone()

    # compute min max of source positions and normalize
    pos_min = torch.amin(pos_orig, dim=(0, 1))
    pos_max = torch.amax(pos_orig, dim=(0, 1))

    # normalize the positions to 0-1
    pos_orig = (pos_orig - pos_min) / (pos_max - pos_min)
    pos_pred = (pos_pred - pos_min) / (pos_max - pos_min)
    pos_orig[torch.isnan(pos_orig)] = 0
    pos_orig[torch.isinf(pos_orig)] = 0
    pos_pred[torch.isnan(pos_pred)] = 0
    pos_pred[torch.isinf(pos_pred)] = 0

    # compute position loss for all positions, not only root
    cycle_loss_positions = cycle_loss_modules[1](pos_orig, pos_pred)


    # ### end effector-related loss
    pos_ret = pos_ret[new_indices]
    pos_orig_v2 = pos_orig_v2[:,:,1:,:]
    pos_ret = pos_ret[:,:,1:,:]
    pos_orig_v2 = torch.div(pos_orig_v2, h_orig)
    pos_ret = torch.div(pos_ret, h_target)
    # tmp = torch.cat( [ pos_orig_v2, pos_ret], dim=0)
    # pos_min = torch.amin(tmp, dim=(0, 1))
    # pos_max = torch.amax(tmp, dim=(0, 1))
    pos_min = torch.amin(pos_orig_v2, dim=(0, 1))
    pos_max = torch.amax(pos_orig_v2, dim=(0, 1))
    pos_orig_v2 = (pos_orig_v2 - pos_min) / (pos_max - pos_min)
    pos_ret = (pos_ret - pos_min) / (pos_max - pos_min)
    pos_orig_v2[torch.isnan(pos_orig_v2)] = 0
    pos_orig_v2[torch.isinf(pos_orig_v2)] = 0
    pos_ret[torch.isnan(pos_ret)] = 0
    pos_ret[torch.isinf(pos_ret)] = 0


    # compute velocities of end effectors only
    vel_orig = pos_orig_v2[:, 1:, :, :] - pos_orig_v2[:, :-1, :, :]
    vel_ret = pos_ret[:, 1:, :, :] - pos_ret[:, :-1, :, :]


    # 3) compute latent loss: the motion representation should be similar
    cycle_loss_latent = cycle_loss_modules[2](motion_reps1, motion_reps2)

    # 4) compute velocity(time_loss) for positions and rotations
    cycle_loss_rotations_time = compute_time_loss( time_loss_module=cycle_loss_modules[3], expected_seq=rotations_original, prediction=rotations_predicted )
    cycle_loss_positions_time = compute_time_loss(time_loss_module=cycle_loss_modules[4],  expected_seq=pos_orig, prediction=pos_pred)
    cycle_loss_time_latent = compute_time_loss(time_loss_module=cycle_loss_modules[5], expected_seq=motion_reps1, prediction=motion_reps2)

    # compute end-effector loss
    end_effector_loss = cycle_loss_modules[6](vel_orig, vel_ret)

    # return all losses
    mult1 = 20
    mult2 = 50
    mult3 = 50
    div = 2

    return [ cycle_loss_rotations, cycle_loss_positions/div, mult1*cycle_loss_latent,
             mult2*cycle_loss_rotations_time,  mult1*cycle_loss_positions_time, mult2*cycle_loss_time_latent, mult3 * end_effector_loss ]

    # no ee loss
    # return [ cycle_loss_rotations, cycle_loss_positions/div, mult1*cycle_loss_latent,
    #          mult2*cycle_loss_rotations_time,  mult1*cycle_loss_positions_time, mult2*cycle_loss_time_latent ]

    # no latent loss
    # return [ cycle_loss_rotations, cycle_loss_positions/div, mult2*cycle_loss_rotations_time,  mult1*cycle_loss_positions_time, mult3 * end_effector_loss ]

    # no time loss
    # return [ cycle_loss_rotations, cycle_loss_positions/div, mult1*cycle_loss_latent, mult3 * end_effector_loss ]


