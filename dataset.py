import torch
import numpy as np
from PIL import Image
import os
import glob
import random
import tqdm

from torch.utils.data import Dataset
from torchvision import transforms

class Vim2Dataset(Dataset):
    def __init__(self, root_dir, is_val=False, subject_id=1,
                ignore_voxel_reponse = False,
                voxel_response_transform=None, stimuli_transform=None,
                add_noise=False, resize_stimuli=False,
                debug=False, verbose=False):
        """
            Initialised the Vim2Dataset
            Params :
                - root_dir : Root Directory of the dataset
                - is_val : Load the validation set ? Else load the Training set
                - subject_id : Subject ID for voxel response. Can accept [1,2,3]
                - ignore_voxel_reponse : Do not load voxel response (and pass random values) : False
                - voxel_response_transform : pytorch transform to be applied to voxel response
                - stimuli_transform : pytorch transform to be applied to the stimuli
                - add_noise : Either holds False when disabled, or [x,y] denoting the range in which to add noise
                - imsize : Resize stimuli images after loading to save time during training. (default : False)
                - debug : Print debug messages, return random tensors in debug mode
                - verbose : Boolean value indicating if we want to print logs in verbose mode
        """
        self.root_dir = root_dir
        self.is_val=is_val
        self.subject_id = subject_id
        self.ignore_voxel_reponse = ignore_voxel_reponse
        self.voxel_response_transform = voxel_response_transform
        self.stimuli_transform = stimuli_transform
        self.add_noise = add_noise
        self.resize_stimuli = resize_stimuli
        self.debug = debug
        self.verbose = verbose

        self.stimuli_capture_frequency = 15 # 15 Hertz

        if not self.debug:
            self.validate_root_dir()
            assert self.subject_id in [1,2, 3]

            self.load_stimuli()
            if not self.ignore_voxel_reponse:
                self.load_voxel_response()

    def load_stimuli(self):
        if self.is_val:
            self.stimuli_path = os.path.join(self.root_dir, "validation_stimuli.npy")
        else:
            self.stimuli_path = os.path.join(self.root_dir, "training_stimuli.npy")

        if self.verbose:
            print("Loading Stimuli from : ", self.stimuli_path)
        self.stimuli = np.load(self.stimuli_path).astype(np.uint8)

    def load_voxel_response(self):
        if self.is_val:
            self.voxel_response_path = os.path.join(
                self.root_dir,
                "VoxelResponses_subject{}_validation.npz".format(self.subject_id)
                )
        else:
            self.voxel_response_path = os.path.join(
                self.root_dir,
                "VoxelResponses_subject{}_training.npz".format(self.subject_id)
                )

        if self.verbose:
            print("Loading Voxel Responses for Subject {} from : {}".format(\
                self.subject_id, self.voxel_response_path))

        self.voxel_responses = np.load(self.voxel_response_path)["arr_0"]

    def __len__(self):
        if self.debug:
            return 1000
        else:
            return len(self.stimuli)

    def __getitem__(self, idx):
        if self.debug:
            random_stimuli = torch.rand(3, 64, 64)
            random_vortex_response = torch.rand(1, 18, 64, 64)
            return random_stimuli, random_vortex_response

        # Obtain and prepare stimuli
        stimuli = self.stimuli[idx] # C x H x W
        print(stimuli.shape)
        stimuli = Image.fromarray(stimuli.astype('uint8'), mode='RGB')
        print(stimuli.size)

        if self.stimuli_transform:
            stimuli = self.stimuli_transform(stimuli)

        # Respons with the same voxel for all 15 stimuli captured at the timepoint
        if not self.ignore_voxel_reponse:
            voxel_response = self.voxel_responses[int(idx/self.stimuli_capture_frequency)]
            if self.voxel_response_transform:
                voxel_response = self.voxel_response_transform(voxel_response)

            voxel_response = torch.FloatTensor(voxel_response).unsqueeze(0)
            if self.add_noise:
                noise = torch.Tensor(1, 18, 64, 64)
                noise = torch.FloatTensor(1, 18, 64, 64).normal_(
                                                            self.add_noise[0],
                                                            self.add_noise[1]
                                                            )
                voxel_response = torch.add(voxel_response, noise)
        else:
            voxel_response = torch.rand(1, 18, 64, 64)
        # shape : [1, 18, 64, 64] Depth x Width x Height
        # which gets added by the batch_size to finally become :
        # [batch_size, 1, 18, 64, 64]
        # where 1 is the number of channels
        return stimuli, voxel_response

        # im = Image.open(self.files[idx])
        # if self.transform is not None:
        #     im = self.transform(im)
        # label = self.labels[idx]

    def validate_root_dir(self):
        expected_files = [ \
         'validation_stimuli.npy', \
         'training_stimuli.npy', \
         'VoxelResponses_subject1_training.npz', \
         'VoxelResponses_subject1_validation.npz', \
         'VoxelResponses_subject2_training.npz', \
         'VoxelResponses_subject2_validation.npz', \
         'VoxelResponses_subject3_training.npz',\
         'VoxelResponses_subject3_validation.npz']

        for _expected_file in expected_files:
            assert os.path.exists(os.path.join(self.root_dir, _expected_file))

if __name__ == "__main__":

    # stimuli_transforms = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    stimuli_transforms = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    voxel_response_transforms = transforms.Compose([
    ])
    dataset = Vim2Dataset(  "data/vim-2",
                            subject_id=1,
                            is_val=True,
                            stimuli_transform = stimuli_transforms,
                            voxel_response_transform=voxel_response_transforms,
                            ignore_voxel_reponse = True,
                            resize_stimuli=(64, 64),
                            debug=False,
                            verbose=True)

    stimuli, voxel_response = dataset[0]
    print("Voxel Response ", voxel_response.shape, " Stimuli : ", stimuli.shape)
    print("Voxel Response Mean ", voxel_response.mean(), " Stimuli Mean : ", stimuli.mean())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    _idx = 0
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(log_dir="./logs/gan_fMRI__run-1/")
    import torchvision.utils as vutils

    _idx = 0
    for stimuli, voxel_response in data_loader:
        print(_idx, voxel_response.shape, stimuli.shape)
        # std_coeff = torch.Tensor(stimuli.shape).fill_(0.5)
        # mean_coeff = torch.Tensor(stimuli.shape).fill_(0.5)
        # unint8_coeff = torch.Tensor(stimuli.shape).fill_(255.0)
        # #
        # real_images = torch.mul(stimuli, std_coeff)
        # real_images = torch.add(real_images, mean_coeff)
        # real_images = torch.mul(real_images, unint8_coeff)
        #
        real_images = stimuli[0:3]
        images = vutils.make_grid(torch.FloatTensor(real_images), normalize=True, scale_each=False)
        writer.add_image("/debug/image",  images, _idx)
        _idx += 1
    print("Actual Length : ", len(dataset))
