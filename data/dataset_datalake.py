import fcntl
import io
import json
import os
import sys
import shutil
import time
import copy
import numpy as np
import data.dataset_fg as dsf
import data.TextUtils as tu
from PIL import Image

sys.path.append(os.path.join(os.environ['LAYERJOT_HOME'], 'ljcv-pycore'))
from LJCVPyCore.TrainDatasets import DatasetN1MultiModalCrops

NGRAM = 3
NTOPICS = 64
ALLIMAGEINFO = "allimageinfo.json"

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
    LOCK_WAIT = 1  # seconds

    # simple python lock system.  Don't need to worry about
    # performance here.
    def acquire_lock(self, root):
        info_dir = os.path.join(root)
        lock_file = os.path.join(info_dir, "lock.tmpfile")
        lock = open(lock_file, 'w+')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except Exception:
            return None

    def release_lock(self, lock):
        if lock is None:
            return
        fcntl.flock(lock, fcntl.LOCK_UN)

    def read_layerjot(self, root, info_file_name):
        images_and_targets = []
        aux_info = []
        images_info = []
        iid_to_classidx = {}
        image_dir = os.path.join(root, "images")
        with open(os.path.join(root, info_file_name), 'r') as f:
            info = json.load(f)

        with open(os.path.join(root, ALLIMAGEINFO), 'r') as af:
            all_info = json.load(af)

        # overall iid to class index is determined by complete
        # dataset.

        imagename_to_iid = {}
        for entry in all_info:
            iid = entry[0]
            class_idx = int(entry[1])
            iid_to_classidx[iid] = class_idx  # will do multiple times

        # images and targets used determined by the info_file

        for entry in info:
            iid = entry[0]
            class_idx = int(entry[1])
            # dest_path = entry[2]
            image_file_name = entry[3]
            attr = entry[4]

            file_path = os.path.join(image_dir, str(iid), image_file_name)

            image_name, fext = os.path.splitext(image_file_name)
            imagename_to_iid[image_name] = iid
            images_and_targets.append([file_path, class_idx, attr])

        return images_and_targets, iid_to_classidx, imagename_to_iid, \
            aux_info, images_info

    def write_layerjot(self, root, info_file_name, info_list, valid_classes):
        file_dir, file_name = os.path.split(info_file_name)
        info_dir = os.path.join(root, file_dir)

        if not os.path.exists(info_dir):
            os.makedirs(info_dir, exist_ok=True)
        elif not os.path.isdir(info_dir):
            raise Exception("{} exists and is not a directory!".
                            format(info_dir))

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

    def create_all_image_info(self, root, project, token, bucket, dataset,
                              clear_data):
        ds_dl = DatasetN1MultiModalCrops(project, token, bucket, dataset)
        ds_dl.create_datasources()
        n1_crops_tbl = \
            ds_dl.images.to_table(columns=['item_id',
                                           'image_name',
                                           'in_fence',
                                           'dims',
                                           'description',
                                           'crop_data']).to_pandas()

        item_ids = n1_crops_tbl['item_id']
        in_fences = n1_crops_tbl['in_fence']
        dims = n1_crops_tbl['dims']
        descs = n1_crops_tbl['description']
        image_names = n1_crops_tbl['image_name']
        crop_data = n1_crops_tbl['crop_data']
        iid_to_classidx = {}

        class_idx = 0
        all_image_info = []
        image_dir = os.path.join(root, "images")
        if os.path.exists(image_dir) and clear_data:
            if not os.path.isdir(image_dir):
                raise Exception("Directory {} is not a directory.  "
                                "Exiting!".format(image_dir))
            shutil.rmtree(image_dir)

        for count in range(len(item_ids)):
            iid = item_ids[count]
            base_dir = os.path.join(image_dir, str(iid))

            # If this is a new iid, assign it a class index
            if iid not in iid_to_classidx:
                class_idx += 1
                iid_to_classidx[iid] = class_idx

                # Create the directory for the iid.  If already exists,
                # just make sure it is a directory.
                if os.path.exists(base_dir):
                    if not os.path.isdir(base_dir):
                        raise Exception("Directory {} is not a directory."
                                        " Exiting!".format(base_dir))
                else:
                    os.makedirs(base_dir, exist_ok=True)

            image_name = str(image_names[count])

            # dim = self.ratio_dims(dims[count])
            dim = dims[count]
            in_fence = in_fences[count]
            desc = descs[count]
            cls_idx = iid_to_classidx[iid]

            attr = {
                'iid': iid,
                'image_name': image_name,
                'dim': dim,
                'in_fence': in_fence,
                'desc': desc
            }

            # handle the attributes and image for this entry

            base_image_name, fext = os.path.splitext(image_name)
            if not fext:
                img_file_name = "{}.jpg".format(base_image_name)
            else:
                img_file_name = image_name

            dest_path = os.path.join(base_dir, img_file_name)
            if not os.path.exists(dest_path):
                photo = crop_data[count]
                stream_image = io.BytesIO(photo)
                image = Image.open(stream_image)
                image.save(dest_path)

            # record complete infomation tuple for this entry.
            info = (iid, cls_idx, dest_path, img_file_name, attr)
            all_image_info.append(info)

        return all_image_info

    def create_evaluation_list(self,
                               project, token, bucket, dataset,
                               root, eval_file, eval_ratio,
                               clear_data, reshuffle):
        """
        Create an evaluation dataset with given eval_ratio.
        param clear_data: clear images and rebuild all instance information
        param reshuffle: reshuffle the eval dataset.  It preserves current
        dataset but reshuffles the evaluation dataset contents.
        """
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
            all_info_path = os.path.join(root, ALLIMAGEINFO)
            allinfo_path_exists = os.path.exists(all_info_path)

            eval_file_path = os.path.join(root, eval_file)
            print("Checking if file {} exists...".format(eval_file_path))
            path_exists = os.path.exists(eval_file_path)

            all_image_info = []
            if not allinfo_path_exists or clear_data:
                if not allinfo_path_exists:
                    print("file {} does not exist; making...".
                          format(all_info_path))
                else:
                    print("Remaking file {}...".format(allinfo_path_exists))

                all_image_info = \
                    self.create_all_image_info(root, project, token,
                                               bucket, dataset,
                                               clear_data)

                # Write the info file
                self.write_layerjot(root, all_info_path, all_image_info, None)
            elif reshuffle:
                # Reshuffle test set
                print("Reshuffling test data set...")
                with open(os.path.join(all_info_path), 'r') as af:
                    all_image_info = json.load(af)
            elif not path_exists:
                with open(all_info_path, 'r') as af:
                    all_image_info = json.load(af)
                print("file {} does not exist; making...".
                      format(eval_file_path))
            else:
                # No action required - all info and eval files exist
                pass

            # if we are either creating, recreating (clear_data),
            # or reshuffling, rewrite eval file path

            if all_image_info:
                classes = set(entry[0] for entry in all_image_info)
                nclasses = len(classes)

                # Now only use eval_ratio of all images for eval.
                np.random.shuffle(all_image_info)
                split_idx = int(len(all_image_info)*eval_ratio)
                eval_list = all_image_info[:split_idx]

                print("N classes {} n eval {}".
                      format(nclasses, len(eval_list)))

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
                                         clear_data, reshuffle):
        """
        Create a new training and evaluation datasets with given eval_ratio.
        param clear_data: clear images and rebuild all instance information
        param reshuffle: reshuffle training and evaluation datasets.
        It preserves current dataset but reshuffles the
        training/eval dataset contents.
        """
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
            all_image_info = []

            all_info_path = os.path.join(root, ALLIMAGEINFO)
            allinfo_path_exists = os.path.exists(all_info_path)
            train_file_path = os.path.join(root, train_file)
            eval_file_path = os.path.join(root, val_file)

            print("Checking if file {} and {} exists...".
                  format(all_info_path, train_file_path))
            path_exists = os.path.exists(train_file_path)

            all_image_info = []
            if not allinfo_path_exists or clear_data:
                if not allinfo_path_exists:
                    print("file {} does not exist; making...".
                          format(all_info_path))
                else:
                    print("Remaking files {}, {} and {}...".
                          format(all_info_path, train_file_path,
                                 eval_file_path))

                all_image_info = \
                    self.create_all_image_info(root, project, token,
                                               bucket, dataset,
                                               clear_data)

                # Write the info file
                self.write_layerjot(root, all_info_path, all_image_info, None)
            elif reshuffle:
                # Reshuffle test set
                print("Reshuffling training/validation data sets...")
                with open(os.path.join(all_info_path), 'r') as af:
                    all_image_info = json.load(af)
            elif not path_exists:
                with open(os.path.join(root, all_info_path), 'r') as af:
                    all_image_info = json.load(af)
                print("files {} and {} do not exist; making...".
                      format(train_file_path, eval_file_path))
            else:
                # No action required - all files exist
                pass

            if all_image_info:
                classes = set(entry[0] for entry in all_image_info)
                nclasses = len(classes)

                # Now partition into training and validation
                np.random.shuffle(all_image_info)
                split_idx = int(len(all_image_info)*training_ratio)
                training_list = all_image_info[:split_idx]
                validation_list = all_image_info[split_idx:]

                print("N classes {} n training {} n val {}".
                      format(nclasses,
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
        images_and_targets, iid_to_classidx, imagename_to_iid, \
            aux_info, images_info = self.read_layerjot(root, info_file)
        return images_and_targets, iid_to_classidx, imagename_to_iid, \
            images_info

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
            clear_data=False,
            reshuffle=False,
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
                                                  clear_data, reshuffle)
            use_file = train_file if train else val_file
        else:
            use_file = os.path.join("eval", "imageinfo.json")
            self.create_evaluation_list(project, token, bucket, dataset,
                                        root, use_file, class_ratio,
                                        clear_data, reshuffle)
        self.aux_info = use_attr or use_txt
        self.dataset = dataset
        self.root = root

        images, iid_to_classidx, imagename_to_iid, images_info = \
            self.extract_images(root, use_file)

        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = iid_to_classidx
        self.imagename_to_iid = imagename_to_iid
        self.images_info = images_info
        self.load_bytes = load_bytes
        self.transform = transform
        self.use_attr = use_attr
        self.use_txt = use_txt
        if use_txt:
            docs = []
            for image_path, class_idx, attr in self.samples:
                desc = attr['desc']
                next_doc = tu.create_ngrams(desc.split(), ngram=NGRAM)
                docs.append(next_doc)
            self.desc_dict, self.desc_lda, self.desc_simindex = \
                tu.create_ldasim(docs, ntopics=NTOPICS)
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
                                         ngram=NGRAM)
            txt_attr = self.vector_lda_to_attr(vec_lda, NTOPICS).tolist()

        return (v_attr, txt_attr)

    def __getitem__(self, index):
        img_path, class_idx, attr = self.samples[index]
        img = open(img_path, 'rb').read() if self.load_bytes \
            else Image.open(img_path).convert('RGB')

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
