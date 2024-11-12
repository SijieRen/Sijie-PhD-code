import torch

def save_checkpoint(state, name, epoch, aim_epoch, is_best, time):


    if is_best:
        aucfilename =  name + str(time) + '_' + 'best_acc.pth.tar'# + str(epoch)
        print('start saving model')
        torch.save(state, aucfilename)
    # if epoch + 1 == aim_epoch:
    #     aucfilename = name + '_' + str(epoch) + 'final_AUC.pth.tar'
    #     print('start saving final model')
    #     torch.save(state, aucfilename)