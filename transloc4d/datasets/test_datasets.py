import torch
import numpy as np
import os
import pickle
import numpy as np
import torch.utils.data as data
import MinkowskiEngine as ME

from .augmentation import ValSetTransform
from .pc_loader import Rad4DPointCloudLoader


class WholeDataset(data.Dataset):
    def __init__(
        self,
        database_pickle,
        query_pickle,
        input_transform=ValSetTransform(aug_mode=1),
        set_transform=None,
        split=None,
        test_split=None,
        mode="test",
    ):
        super().__init__()
        self.split = split
        self.pc_loader = Rad4DPointCloudLoader()
        self.database = self.load_pickle(database_pickle)
        self.queries = self.load_pickle(query_pickle)
        self.construct_quries(self.queries)
        self.construct_database(self.database)
        self.wholedataset_pc_file = self.queries_pc_file + self.database_pc_file
        self.len_q = len(self.queries_pc_file)
        self.len_db = len(self.database_pc_file)
        self.input_transform = input_transform
        # pc_loader must be set in the inheriting class
        self.set_transform = set_transform
        self.mode = mode

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def construct_quries(self, df_query):
        self.queries = {}
        self.queries_id = []
        self.queries_pc_file = []
        self.queries_nonnegatives = []
        for anchor_ndx in range(len(df_query)):
            non_negatives = df_query[anchor_ndx]["positives"]
            if len(non_negatives) == 0:
                continue

            # Extract timestamp from the filename
            scan_filename = df_query[anchor_ndx]["file"]
            assert os.path.isfile(scan_filename), "point cloud file {} is found".format(
                scan_filename
            )

            # Sort ascending order
            non_negatives = np.sort(non_negatives)

            self.queries_id.append(anchor_ndx)
            self.queries_pc_file.append(scan_filename)
            # self.queries_positives.append(positives)
            self.queries_nonnegatives.append(non_negatives)

        print(f"==> Queries: {len(self.queries_nonnegatives)} valid queries")

    def construct_database(self, df_db):
        self.database_id = []
        self.database_pc_file = []

        for idx in range(len(df_db)):
            self.database_id.append(idx)
            scan_filename = df_db[idx]["file"]
            self.database_pc_file.append(scan_filename)

        print(f"==> Database: {len(self.database_pc_file)} references")

    def __len__(self):
        return len(self.wholedataset_pc_file)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        file_pathname = self.wholedataset_pc_file[ndx]
        query_pc = self.pc_loader(file_pathname)
        query_pc = torch.tensor(query_pc, dtype=torch.float)
        if self.input_transform is not None:
            query_pc = self.input_transform(query_pc)
        return query_pc, ndx

    def get_path(self, ndx):
        return self.wholedataset_pc_file[ndx]

    def get_non_negatives(self, ndx):
        return self.queries_nonnegatives[ndx]

def resample_tensor(input_tensor, output_shape=4096):
    input_tensor = input_tensor.transpose(0, 1).unsqueeze(0)

    output_tensor = torch.nn.functional.interpolate(
        input_tensor, size=output_shape, mode="nearest"
    )

    output_tensor = output_tensor.squeeze(0).transpose(0, 1).contiguous()

    return output_tensor

def test_collate_fn(
    dataset,
    quantizer,
    batch_split_size=None,
    input_representation="RVI",
    scancontext_input=False,
):
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]

        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.transform(clouds)
            clouds = clouds.split(lens)

        clouds_org_coords = [e[:, :3] for e in clouds]
        coords_quant = (
            [quantizer(e)[0]
             for e in clouds] if quantizer is not None else clouds
        )
        coords = [e[:, :3] for e in coords_quant]
        if input_representation == "RV":
            feats = [e[:, 3:4] for e in coords_quant]
        elif input_representation == "RI":
            feats = [e[:, 4:] for e in coords_quant]
        else:
            feats = [e[:, 3:] for e in coords_quant]

        if batch_split_size is None or batch_split_size == 0:
            c = (
                ME.utils.batched_coordinates(coords)
                if not scancontext_input
                else ME.utils.batched_coordinates(coords, dtype=torch.float32)
            )
            # Assign a dummy feature equal to 1 to each point
            if input_representation == "R":
                feats = torch.ones((c.shape[0], 1), dtype=torch.float32)
            else:
                feats = torch.cat(feats, 0)
            batch = {
                "coords": c,
                "features": feats,
                "batch": torch.stack(clouds_org_coords),
            }

        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp_org = clouds_org_coords[i: i + batch_split_size]
                temp = coords[i: i + batch_split_size]
                c = (
                    ME.utils.batched_coordinates(temp)
                    if not scancontext_input
                    else ME.utils.batched_coordinates(temp, dtype=torch.float32)
                )
                if input_representation == "R":
                    f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                else:
                    f = torch.cat(feats[i: i + batch_split_size], 0)
                minibatch = {"coords": c, "features": f,
                             "batch": torch.stack(temp_org)}
                batch.append(minibatch)

        return batch

    return collate_fn
