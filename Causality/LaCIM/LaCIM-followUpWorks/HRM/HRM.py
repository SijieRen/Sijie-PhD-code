from .Backend import MpModel, pretty
from .Frontend import McModel
from .Selection_bias import Multi_env_selection_bias, generate_test, modified_Multi_env_selection_bias
import torch
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from .utils_baseline import get_opt


def compute_acc(pred, target):
    return (np.sum(np.argmax(pred, axis=1) == target).astype('int')) / pred.shape[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HRM:
    def __init__(self, front_params, back_params, args=None):
        self.args = args
        # self.X = X
        # target = y
        # self.test_X = [test_X]
        # self.test_y = [test_y]
        self.frontend = McModel(front_params['num_clusters'], args)
        self.backend = MpModel(input_dim=back_params['input_dim'],
                               output_dim=back_params['output_dim'],
                               sigma=back_params['sigma'],
                               lam=back_params['lam'],
                               alpha=back_params['alpha'],
                               hard_sum=back_params['hard_sum'],
                               penalty="Ours",
                               args=args)
        self.domains = None
        if 'NICO' in self.args.dataset:
            self.weight = torch.tensor(
                (3, 256, 256), dtype=torch.float32).cuda()
        if 'mnist' in self.args.dataset:
            self.weight = torch.tensor((3, 28, 28), dtype=torch.float32).cuda()
        if 'AD' in self.args.dataset:
            self.weight = torch.tensor(
                (48, 48, 48), dtype=torch.float32).cuda()

    def solve(self, iters, trainloader):
        self.density_result = None
        density_record = []
        flag = False
        for i in range(iters):
            print("*** McModel ***")
            train_Mc_sequence = 0
            for batch_idx, (x, u, us, target) in enumerate(trainloader):
                print('McModel Train Process: [{}/{} ({:.0f}%)] '.format(
                    train_Mc_sequence, len(trainloader.dataset),
                    100. * train_Mc_sequence / len(trainloader.dataset)))
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()
            # if self.args.cuda:
                self.domains = self.frontend.cluster(
                    self.weight, self.domains, x, target, flag)
                train_Mc_sequence += len(x)
            print("*** MpModel ***")
            train_Mp_sequence = 0

            for batch_idx, (x, u, us, target) in enumerate(trainloader):
                print('MpModel Train Process: [{}/{} ({:.0f}%)] '.format(
                    train_Mp_sequence, len(trainloader.dataset),
                    100. * train_Mp_sequence / len(trainloader.dataset)))
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()
                for i in range(self.args.env_num):
                    index = torch.where(self.domains == i)[0]
                    tempx = (x[index, :])  # .reshape(-1, self.shape_1)
                    tempy = (target[index])  # .reshape(-1, 1)
                    if not tempx.size(0) == 0:
                        weight, density = self.backend.train(
                            tempx, tempy, epochs=self.args.epochs)
                train_Mp_sequence += len(x)
            density_record.append(density)
            self.density_result = density
            self.weight = density
            self.backend.lam *= 1.05
            self.backend.alpha *= 1.05
            print('Selection Ratio is %s' % self.weight)

        f = open('./save.txt', 'a+')
        print('Density results:')
        for i in range(len(density_record)):
            print("Iter %d Density %s" % (i, pretty(density_record[i])))
            f.writelines(pretty(density_record[i]) + '\n')
        f.close()
        # print("self.weight: ", self.weight.size())
        return self.weight

    def test(self, testloader, mask, treshold):  # TODO test xuyao xiugai
        accuracy = AverageMeter()
        pred = np.zeros((testloader.dataset.__len__(), self.args.num_classes))
        test_accs = []
        self.backend.backmodel.eval()
        self.backend.featureSelector.eval()
        batch_begin = 0
        mask = mask
        for batch_idx, (x, u, us, target) in enumerate(testloader):
            if self.args.cuda:
                # if 'mnist' in args.dataset or 'NICO' in args.dataset:
                #     x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()
                # else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()
            # for i in range(len(test_envs)):
                # print(mask.size())
                # print("x", x.size())
                # print("mask", len(mask), mask[0].size())
                # print("x mask: ", x[:, mask > treshold],
                #       x[:, mask > treshold].size())
                if self.args.dataset == 'AD':
                    pred_y = self.backend.single_forward(
                        x[:, :, mask > treshold].view(x.size(0), 1, 48, 48, 48))
                if 'NICO' in self.args.dataset:
                    pred_y = self.backend.single_forward(
                        x[:, mask > treshold].view(x.size(0), 3, 256, 256))
                if 'mnist' in self.args.dataset:
                    pred_y = self.backend.single_forward(
                        x[:, mask > treshold].view(x.size(0), 3, 28, 28))

                # error = torch.sqrt(torch.mean(
                #     (pred.reshape(test_envs[i][1].shape) - test_envs[i][1]) ** 2))
                # test_accs.append(error.data)
            pred[batch_begin:batch_begin +
                 x.size(0), :] = pred_y.detach().cpu()
            batch_begin = batch_begin + x.size(0)
            accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                                        target.detach().cpu().numpy()), x.size(0))
            # print(pretty(test_accs))

        self.backend.backmodel.train()
        self.backend.featureSelector.train()
        return pred, accuracy.avg


def combine_envs(envs):
    X = []
    y = []
    for env in envs:
        X.append(env[0])
        y.append(env[1])
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    return X.reshape(-1, X.shape[1]), y.reshape(-1, 1)


def seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class EmpiricalRiskMinimizer(object):
    def __init__(self, X, y, mask):
        x_all = X.numpy()
        y_all = y.numpy()
        self.mask = mask
        x_all = x_all[:, self.mask]
        w = LinearRegression(fit_intercept=False).fit(x_all, y_all).coef_
        self.w = torch.Tensor(w)

    def solution(self):
        return self.w

    def test(self, X, y):
        X = X.numpy()
        X = X[:, self.mask]
        y = y.numpy()
        err = np.mean((X.dot(self.w.T) - y) ** 2.).item()
        return np.sqrt(err)


def train(model, ):
    pass


if __name__ == "__main__":
    all_weights = torch.tensor(np.zeros(10, dtype=np.float32))
    average = 0.0
    std = 0.0
    seeds = 10
    average_error_list = torch.Tensor(np.zeros(10, dtype=np.float))
    for seed in range(0, seeds):
        seed_torch(seed)
        print("---------------seed = %d------------------" % seed)
        environments, _ = Multi_env_selection_bias()
        X, y = combine_envs(environments)
        args = get_opt()
        # params
        front_params = {}
        front_params['num_clusters'] = args.env_num

        back_params = {}
        back_params['input_dim'] = X[0].size()
        back_params['output_dim'] = 1
        back_params['sigma'] = 0.1
        back_params['lam'] = 0.1
        back_params['alpha'] = 1000.0
        back_params['hard_sum'] = 10
        back_params['overall_threshold'] = 0.20
        whole_iters = 5

        # train and test
        model = HRM(front_params, back_params, args)
        result_weight = model.solve(whole_iters, X, y)
        all_weights += result_weight

        mask = torch.where(result_weight > back_params['overall_threshold'])[0]
        evaluate_model = EmpiricalRiskMinimizer(X, y, mask)
        testing_envs = generate_test()

        testing_errors = []
        for [X, y] in testing_envs:
            testing_errors.append(evaluate_model.test(X, y))

        testing_errors = torch.Tensor(testing_errors)
        print(testing_errors)
        average += torch.mean(testing_errors) / seeds
        std += torch.std(testing_errors) / seeds
        average_error_list += testing_errors / seeds
        print(average_error_list)

    print(average)
    print(std)
    print(all_weights)
