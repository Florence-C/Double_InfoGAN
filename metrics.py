
import os
import numpy as np

from torchvision.utils import make_grid, save_image
import torch


class FactorVAEMetricDouble:
    """ Impementation of the metric in: 
        Disentangling by Factorising
    """
    def __init__(self, metric_data_groups, metric_data_eval_std, cuda, *args, **kwargs):
        super(FactorVAEMetricDouble, self).__init__(*args, **kwargs)
        self.metric_data_groups = metric_data_groups
        self.metric_data_eval_std = metric_data_eval_std

        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        print("in init FactorVAEMetricDouble")

    def evaluate(self, discriminator=None):

        #print("in evaluate")
        #print(self.metric_data["img_eval_std"].shape)
        
        # eval_std_inference = self.model.inference_from(
        #     self.metric_data["img_eval_std"])
        # eval_std = np.std(eval_std_inference, axis=0, keepdims=True)

        _, _, _ , eval_std_inference = discriminator(self.FloatTensor(self.metric_data_eval_std[:1000]*2-1))
        eval_std = np.std(eval_std_inference.detach().cpu().numpy(), axis=0, keepdims=True)
        #eval_std = 1
        print('eval_std = ', eval_std)
        labels = set(data["label"] for data in self.metric_data_groups)

        train_data = np.zeros((len(labels), len(labels)))

        for data in self.metric_data_groups:

            #print(data["img"].shape)

            #print(data["img"]*2 - 1)

            _, _, _, data_inference = discriminator(self.FloatTensor(data["img"]*2-1))

            data_inference_d = data_inference.detach().cpu().numpy()
            data_inference_d /= eval_std
            data_std = np.std(data_inference_d, axis=0)
            predict = np.argmin(data_std)
            train_data[predict, data["label"]] += 1

            # predict = (data["label"] + 1) % 5
            # train_data[predict, data["label"]] += 1

        total_sample = np.sum(train_data)
        maxs = np.amax(train_data, axis=1)
        correct_sample = np.sum(maxs)

        correct_sample_revised = np.flip(np.sort(maxs), axis=0)
        correct_sample_revised = np.sum(
            correct_sample_revised[0: train_data.shape[1]])

        return {"factorVAE_metric": float(correct_sample) / total_sample,
                "factorVAE_metric_revised": (float(correct_sample_revised) /
                                             total_sample),
                "factorVAE_metric_detail": train_data}



if __name__ == "__main__":


    metric_data_groups = np.load("metric_data_groups_gridmnist_dsprite.npy", allow_pickle=True)
    metric_data_eval_std = np.load("metric_data_eval_std.npy", allow_pickle=True) 


    #print("groups = ", metric_data["groups"])

    i = 0
    for data in metric_data_groups: 
        if i ==0 : 
            print(data["img"].shape)

            save_image(torch.tensor(data["img"][0]), 'test_dataset_0.jpg', normalize=True)
            save_image(torch.tensor(data["img"][1]), 'test_dataset_1.jpg', normalize=True)
            save_image(torch.tensor(data["img"][2]), 'test_dataset_2.jpg', normalize=True)
            save_image(torch.tensor(data["img"][3]), 'test_dataset_3.jpg', normalize=True)
            save_image(torch.tensor(data["img"][4]), 'test_dataset_4.jpg', normalize=True)

        i+=1


    print('nb data = ', i)


    i = 0
    for data in metric_data_eval_std: 
        if i ==0 : 
            print(data)
            # print(data["img"].shape)

            # save_image(torch.tensor(data["img"][0]), 'std_img_0.jpg', normalize=True)
            # save_image(torch.tensor(data["img"][1]), 'std_img_1.jpg', normalize=True)
            # save_image(torch.tensor(data["img"][2]), 'std_img_2.jpg', normalize=True)
            # save_image(torch.tensor(data["img"][3]), 'test_dataset_3.jpg', normalize=True)
            # save_image(torch.tensor(data["img"][4]), 'test_dataset_4.jpg', normalize=True)

        i+=1

    print('nb eval = ', i)

    fvaem = FactorVAEMetricDouble(metric_data_groups,metric_data_eval_std, True)

    #print("metric_data = ", metric_data)

    metric = fvaem.evaluate()

    print(metric)
