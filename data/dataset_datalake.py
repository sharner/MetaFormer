import data.dataset_fg as dsf
import data.TextUtils as tu
from PIL import Image
import fcntl
import io
import json
import numpy as np
import os
import sys
import shutil
import random
import time
import copy

NGRAM = 3
NTOPICS = 64

sys.path.append(os.path.join(os.environ['LAYERJOT_HOME'], 'ljcv-pycore'))
from LJCVPyCore.TrainDatasets import DatasetN1Crops, DatasetN1MultiModalCrops

# Required to properly encode numpy data types.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DatasetMeta_dl(dsf.DatasetMeta):
    LOCK_WAIT = 1 # seconds

    # simple python lock system.  Don't need to worry about
    # performance here.
    def acquire_lock(self, root):
        info_dir = os.path.join(root)
        lock_file = os.path.join(info_dir, "lock.tmpfile")
        lock = open(lock_file, 'w+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except:
            return None

    def release_lock(self, lock):
        if lock is None:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)

    def read_layerjot(self, root, info_file_name):
        images_and_targets = []
        aux_info = []
        images_info = []
        class_to_iid = {}
        image_dir = os.path.join(root, "images")
        with open(os.path.join(root, info_file_name), 'r') as f:
            info = json.load(f)

        for entry in info:
            iid = entry[0]
            class_idx = int(entry[1])
            dest_path = entry[2]
            image_name = entry[3]
            attr = entry[4]
    
            file_path = os.path.join(image_dir, str(iid), image_name)
            class_to_iid[class_idx] = iid # will do multiple times
            images_and_targets.append([file_path, class_idx, attr])

        return images_and_targets, class_to_iid, aux_info, images_info

    def write_layerjot(self, root, info_file_name, info_list, valid_classes):
        file_dir, file_name = os.path.split(info_file_name)
        info_dir = os.path.join(root, file_dir)

        if not os.path.exists(info_dir):
            os.makedirs(info_dir, exist_ok=True)
        elif not os.path.isdir(info_dir):
            raise Exception("{} exists and is not a directory!".format(info_dir))

        info_file_path = os.path.join(info_dir, file_name)

        # prune the list if valid_classes is set
        if valid_classes:
            pruned_list = []
            for info in info_list:
                if info[1] in valid_classes:
                    pruned_list.append(info)

            info_list = pruned_list

        with open(info_file_path, 'w') as f:
            json.dump(info_list, f, cls=NpEncoder)
            print("write completed")
            
    def ratio_dims(self, dims):
        mod_dims = copy.deepcopy(dims)
        if len(mod_dims) < 2:
            return mod_dims
        d1 = float(mod_dims[0])
        if d1 < 1.0e-08:
            mod_dims[0] = float(mod_dims[1])
            mod_dims[1] = 0
        else:
            mod_dims[1] = mod_dims[1]/d1
        return mod_dims

    def create_all_image_info(self, root, project, token, bucket, dataset, clear_dir, overwrite):
        ds_dl = DatasetN1MultiModalCrops(project, token, bucket, dataset)
        ds_dl.create_datasources()
        n1_crops_tbl = ds_dl.images.to_table(columns=['item_id', 'image_name', 'in_fence', 'dims', 'description','crop_data']).to_pandas()

        item_ids = n1_crops_tbl['item_id']
        in_fences = n1_crops_tbl['in_fence']
        dims = n1_crops_tbl['dims']
        descs = n1_crops_tbl['description']
        image_names = n1_crops_tbl['image_name']
        crop_data = n1_crops_tbl['crop_data']

        class_idx = 0
        all_image_info = []
        image_dir = os.path.join(root, "images")
        if os.path.exists(image_dir) and clear_dir:
            if not os.path.isdir(image_dir):
                raise Exception("Directory {} is not a directory.  "
                                "Exiting!".format(image_dir))
            shutil.rmtree(image_dir)
                    
        iid_to_classidx = {}
        for count in range(len(item_ids)):
            iid = item_ids[count]
            base_dir = os.path.join(image_dir, str(iid))

            # If this is a new iid, assign it a class index
            if not iid in iid_to_classidx:
                iid_to_classidx[iid] = class_idx
                class_idx += 1
 
                # Create the directory for the iid.  If already exists,
                # just make sure it is a directory.
                if os.path.exists(base_dir):
                    if not os.path.isdir(base_dir):
                        raise Exception("Directory {} is not a directory."
                                        " Exiting!".format(base_dir))
                    else:
                        os.makedirs(base_dir, exist_ok=True)

            image_name = str(image_names[count])
            #dim = self.ratio_dims(dims[count])
            dim = dims[count]
            in_fence = in_fences[count]
            desc = descs[count]
            cls_idx = iid_to_classidx[iid]

            attr = {
                'iid' : iid,
                'image_name' : image_name,
                'dim' : dim,
                'in_fence' : in_fence,
                'desc' : desc
            }

            # handle the attributes and image for this entry
                    
            img_file_name = "{}.jpg".format(image_name)
            dest_path = os.path.join(base_dir, img_file_name)
            if not os.path.exists(dest_path) or overwrite:
                photo = crop_data[count]
                stream_image = io.BytesIO(photo)
                image = Image.open(stream_image)
                image.save(dest_path)

            # record complete infomation tuple for this entry.
            info = (iid, cls_idx, dest_path, img_file_name, attr)
            all_image_info.append(info)
        return all_image_insfo
    
    def create_evaluation_list(self,
                               project, token, bucket, dataset,
                               root, eval_file,
                               eval_ratio=0.10,
                               overwrite=True,
                               clear_dir=True):
        # Wait for lock.  Once we have lock, we either read the
        # information directly from the training/validation directory
        # or we create those two files
        while True:
            lock = self.acquire_lock(root)
            if lock:
                break
            time.sleep(self.LOCK_WAIT)

        # The critical section is writing eval file

        try:
            eval_file_path = os.path.join(root, eval_file)
            print("Checking if file {} exists...".format(eval_file_path))
            if not os.path.exists(eval_file_path):
                print("file {} does not exist. Making...".format(eval_file_path))
                all_image_info = self.create_all_image_info(root, project, token, bucket, dataset, clear_dir, overwrite)
                                   
                # Now only use eval_ratio of all images for eval.
                np.random.shuffle(all_image_info)
                split_idx = int(len(all_image_info)*eval_ratio)
                eval_list = all_image_info[:split_idx]

                print("N classes {} n eval {}".
                      format(class_idx+1, len(eval_list)))

                # write info into training and validation file
                self.write_layerjot(root, eval_file_path, eval_list, None)
        finally:
            # once the information is written, we can
            # release the lock
            self.release_lock(lock)

    def create_training_validation_lists(self,
                                         project, token, bucket, dataset,
                                         root, training_ratio,
                                         train_file, val_file,
                                         overwrite=True,
                                         clear_dir=True):
        # Wait for lock.  Once we have lock, we either read the
        # information directly from the training/validation directory
        # or we create those two files
        while True:
            lock = self.acquire_lock(root)
            if lock:
                break
            time.sleep(self.LOCK_WAIT)

        # The critical section is writing the training and validation files.
        # If they do not exist, then create them

        try:
            train_file_path = os.path.join(root, train_file)
            print("Checking if file {} exists...".format(train_file_path))
            if not os.path.exists(train_file_path):
                print("file {} does not exist. Making...".format(train_file_path))
                all_image_info = self.create_all_image_info(root, project, token, bucket, dataset, clear_dir, overwrite)
                                   
                # Now partition into training and validation
                np.random.shuffle(all_image_info)
                split_idx = int(len(all_image_info)*training_ratio)
                training_list = all_image_info[:split_idx]
                validation_list = all_image_info[split_idx:]

                print("N classes {} n training {} n val {}".
                      format(class_idx+1,
                             len(training_list),
                             len(validation_list)))

                # determine classes that have training examples
                valid_train_classes = set()
                for train_info in training_list:
                    cidx = train_info[1]
                    valid_train_classes.add(cidx)

                # write info into training and validation file
                self.write_layerjot(root, train_file, training_list, None)
                self.write_layerjot(root, val_file, validation_list,
                                    valid_train_classes)
        finally:
            # once the information is written, we can
            # release the lock
            self.release_lock(lock)

    def extract_images(self, root, info_file):
        images_and_targets, class_to_iid, aux_info, images_info = \
                self.read_layerjot(root, info_file)
        return images_and_targets, class_to_iid, images_info
    
    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            train=False,
            aux_info=False,
            dataset='n1_multimodal_data',
            project=None,
            bucket='ljcv-datalake-dev',
            token='google_default',
            class_ratio=1.0,
            per_sample=1.0,
            overwrite=True,
            clear_dir=True,
            use_attr=False,
            use_txt=False,
            eval_mode=False
    ):
        if not eval_mode:
            train_file = os.path.join("train", "imageinfo.json")
            val_file = os.path.join("val", "imageinfo.json")
            self.create_training_validation_lists(project, token, bucket,
                                                  dataset, root, class_ratio,
                                                  train_file, val_file,
                                                  overwrite=overwrite,
                                                  clear_dir=clear_dir)
            use_file = train_file if train else val_file
        else:
            use_file = os.path.join("eval", "imageinfo.json")
            self.create_evaluation_list(project, token, bucket, dataset,
                                        root, use_file, class_ratio,
                                        overwrite=overwrite,
                                        clear_dir=clear_dir)
        self.aux_info = use_attr or use_txt
        self.dataset = dataset
        self.root = root

        images, class_to_iid, images_info = self.extract_images(root, use_file)

        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_iid
        self.images_info = images_info
        self.load_bytes = load_bytes
        self.transform = transform
        self.use_attr = use_attr
        self.use_txt = use_txt
        if use_txt:
            docs = []
            for image_path, class_idx, attr in self.samples:
                desc = attr['desc']
                next_doc = tu.create_ngrams(desc.split(), ngram = NGRAM)
                docs.append(next_doc)
            self.desc_dict, self.desc_lda, self.desc_simindex = \
                tu.create_ldasim(docs, ntopics = NTOPICS)
        else:
            self.desc_dict = self.desc_lda = self.desc_simindex = None

    def vector_lda_to_attr(self, vec_lda, ntopics):
        attr = np.zeros(ntopics)
        for idx, score in vec_lda:
            attr[idx] = score
        return attr

    def attr_to_metainfo(self, attr):
        v_attr = []
        txt_attr = []
        if self.use_attr:
            in_fence = attr['in_fence']
            dims = attr['dim']
            for d in dims:
                v_attr.append(d)
            v_attr.append(in_fence)

        if self.use_txt:
            desc = attr['desc']
            vec_lda, sims = tu.lda_score(desc,
                                         self.desc_dict,
                                         self.desc_lda,
                                         self.desc_simindex,
                                         ngram = NGRAM)
            txt_attr = self.vector_lda_to_attr(vec_lda, NTOPICS).tolist()

        return (v_attr, txt_attr)
            
    def __getitem__(self, index):
        img_path, class_idx, attr = self.samples[index]
        img = open(img_path, 'rb').read() if self.load_bytes else Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.aux_info:
            v_attr, txt_attr = self.attr_to_metainfo(attr)
            all_attr = v_attr + txt_attr
        else:
            all_attr = []


        if all_attr:
            return img, class_idx, np.asarray(all_attr).astype(np.float64)
        else:
            return img, class_idx

