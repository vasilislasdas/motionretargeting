import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    # folder that contains results from training
    folder = "/home/vlasdas/Desktop/thesis/makis_final_ee_loss_long/"

    # move to specified folder
    os.chdir(folder)

    # load all losses
    z = np.load("all_losses.npz")

    #
    disc = z["losses_disc"]
    gen = z["losses_gen"]
    fid = z["losses_valid"]
    sep = z["losses_gen_seperate"]

    # index and value of minimum combined generator loss
    index = np.argmin( gen )
    print(f"index:{index, gen[index]}")
    tmp = sep[:,:-1]
    tmp = np.array(tmp)
    tmp = np.sum(tmp, axis=1)

    # index of everything apart for gen_disc_loss
    index = np.argmin(tmp)
    print(f"index:{index, tmp[index]}")

    # plot fid score
    plt.plot(fid[:,0],fid[:,1])
    plt.title("FID-score")
    plt.xlabel("epochs")
    plt.ylabel("fid")

    #plot total loss generator
    plt.figure()
    # plt.plot(gen, '.')
    plt.plot(tmp, '.')
    plt.title("Generator loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    ax = plt.gca()
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 0.1])

    #plot discriminator loss
    plt.figure()
    disc = disc[:1001] # hack
    plt.plot(disc)
    plt.title("Discriminator loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")





    # plot generator adversarial loss
    plt.figure
    tmp = sep[:, -1]
    tmp = tmp[tmp!=0]
    tmp= tmp[:501]
    plt.plot(tmp)
    plt.title("Generator adversarial loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")


    plt.show()
