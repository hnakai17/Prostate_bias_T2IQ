import pandas as pd
import numpy as np
from monai.transforms import (SpatialPadd, LoadImaged, Resized, ScaleIntensityd,
                              ConcatItemsd, RandFlipd, ThresholdIntensityd, CenterSpatialCropd, RandRotated,
                              RandShiftIntensityd, Rotated, RandGaussianNoised, ScaleIntensityRanged,
                              RandAdjustContrastd, RandZoomd, RandAffined, Compose, EnsureChannelFirstd,
                            Compose, RandSpatialCropd, CropForegroundd)
import torch.utils.data as data
from imblearn.over_sampling import RandomOverSampler
from collections.abc import Callable, Sequence

import numpy as np
import torch
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.functional import crop_func, pad_func
from monai.transforms.croppad.array import Pad
from monai.transforms.croppad.dictionary import Padd, Cropd
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.utils import (
    LazyAttr,
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg_default,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)
from monai.transforms.transform import LazyTransform
from itertools import chain
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    generate_spatial_bounding_box,
    is_positive,
)
from monai.transforms.croppad.array import Crop, BorderPad
from collections.abc import Callable, Hashable, Mapping, Sequence

class ProstateDataset(data.Dataset):
    def __init__(self, img_path, mask_path, label, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        dic = {"img_path":self.img_path[idx],
               "mask_path":self.mask_path[idx],
               "label":self.label[idx]}

        if self.transform:
            return self.transform(dic)
        else:
            return dic

def create_transform_T2(spatial_size, roi_size, set, use_mask=True,
                     center_crop_xy=-1,
                     RandRotated_prob=0.2, RandRotated_range_z=0.3,
                     RandAdjustContrastd_prob=0.1, RandAdjustContrastd_gamma=(0.9, 1.1)):
    # For T2WI.
    # spatial_size: reshaped size (x, x, -1)
    # roi_size: select z (-1, -1, 8 or 16)
    z = roi_size[2]
    if use_mask:
        image_keys = ["img_path", "mask_path"]
        transform_dict_key = f"{set}_with_mask"
    else:
        image_keys = ["img_path"]
        transform_dict_key = f"{set}_wo_mask"

    basics1 = [LoadImaged(keys=image_keys),
               EnsureChannelFirstd(keys=image_keys),
               SpatialPadSquared(keys=image_keys), # convert to square sized matrix  # 100, 50, z -> 100, 100, z
               CropForeground_CustomFOVsized(keys=image_keys, source_key="img_path"), # Crop x, y (110mm , 110mm , z)
               Resized(keys = image_keys, spatial_size=spatial_size),  # 224, 224,  z
              ]

    # Center 8 slice. Training: Center select 10 slice, then randomly pick up 8. Val and test: Center 8 slice.
    training_z_select = [CenterSpatialCropd(keys=image_keys, roi_size=(center_crop_xy, center_crop_xy, z+2)), RandSpatialCropd(keys=image_keys, roi_size=roi_size)] #12, random8
    val_z_select = [CenterSpatialCropd(keys=image_keys, roi_size=roi_size)]

    basics2 = [Resized(keys = image_keys, spatial_size=(-1, -1, 32)) #224, 224, 32
               ] #256, 256, 32

    random_aug = [RandFlipd(keys = image_keys, prob=0.5, spatial_axis=0),
                  RandRotated(keys=image_keys, prob=RandRotated_prob, range_z=RandRotated_range_z, padding_mode="zeros"), # padding_mode should be "zeros" for artifact. (default: border)
                  RandAdjustContrastd(prob=RandAdjustContrastd_prob, gamma=RandAdjustContrastd_gamma, keys = ["img_path"])
                  ]

    img_intensity = [ScaleIntensityd(keys = ["img_path"], channel_wise=True)]
    mask_intensity = [ThresholdIntensityd(keys = ["mask_path"], threshold=0.2, above=False, cval=1)]
    concat = [ConcatItemsd(keys=image_keys, name="img_path", dim=0)]

    train_with_mask_list = basics1 + training_z_select + basics2 + random_aug + img_intensity + mask_intensity + concat
    train_wo_mask_list = basics1 + training_z_select + basics2 + random_aug + img_intensity + concat
    val_with_mask_list = basics1 + val_z_select + basics2 + img_intensity + mask_intensity + concat
    val_wo_mask_list = basics1 + val_z_select + basics2 + img_intensity + concat

    transform_dict = {"train_with_mask":Compose(train_with_mask_list),
                      "train_wo_mask":Compose(train_wo_mask_list),
                      "val_with_mask":Compose(val_with_mask_list),
                      "val_wo_mask":Compose(val_wo_mask_list)}

    return transform_dict[transform_dict_key]

def oversample_clean_train_label(train_img, train_mask, train_categorical_label, clf, num_classes, sampling_strategy="auto"):
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)
    train_img_mask = np.stack([train_img, train_mask], axis=1)
    train_img_mask, train_label = ros.fit_resample(train_img_mask, train_categorical_label)
    train_img, train_mask = train_img_mask[:, 0], train_img_mask[:, 1]
    # Label counts after oversampling
    print("Oversampling of minority class:")
    print(pd.Series(train_categorical_label).value_counts().sort_index())
    print("->")
    print(pd.Series(train_label).value_counts().sort_index())

    train_label =  train_label.astype(dtype = np.float32)
    if clf=="regression":
        train_label = train_label.reshape(-1, 1) # 0, 1, 2, 3 -> 0, 0.33, 0.66, 1
    elif clf=="classification":
        if num_classes==4:
            train_label =  pd.get_dummies(train_label).astype(float).values
        elif num_classes==2:
            train_label = pd.get_dummies(train_label.replace({0:0, 1:0, 2:1, 3:1})).astype(float).values
            train_label = train_label.reshape(-1, 1)
    return train_img, train_mask, train_label

#### Custom MONAI transform ####
class CropForeground_CustomFOVsize(Crop):
    """
    Crop x,y using FOV (default: 120mm)
    """
    @deprecated_arg_default("allow_smaller", old_default=True, new_default=False, since="1.2", replaced="1.5")
    def __init__(
            self,
            select_fn: Callable = is_positive,
            channel_indices: IndexSelection | None = None,
            margin: Sequence[int] | int = 0,
            allow_smaller: bool = True,
            return_coords: bool = False,
            k_divisible: Sequence[int] | int = 1,
            mode: str = PytorchPadMode.CONSTANT,
            lazy: bool = False,
            xy_FOVsize = 120,
            **pad_kwargs,
    ) -> None:

        LazyTransform.__init__(self, lazy)
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.padder = Pad(mode=mode, lazy=lazy, **pad_kwargs)
        self.xy_FOVsize = xy_FOVsize

    @Crop.lazy.setter  # type: ignore
    def lazy(self, _val: bool):
        self._lazy = _val
        self.padder.lazy = _val

    @property
    def requires_current_data(self):
        return False

    """
    def compute_bounding_box(self, img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        #Compute the start points and end points of bounding box to crop.
        #And adjust bounding box coords to be divisible by `k`.
        box_start, box_end = generate_spatial_bounding_box(
            img, self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        print(box_start_, box_end_)
        print(type(box_start_), type(box_end_))
        return box_start_, box_end_
    """

    def compute_bounding_box(self, img: torch.Tensor, xy_FOVsize):
        """
        Custom function to crop image to "xy_size"mm FOV (only x and y dimensions)
        :param image:
        :param xy_size:
        :return:
        """
        _, shape_x, shape_y, shape_z = img.shape
        spacing_x, spacing_y, spacing_z = img.pixdim

        target_x = int(xy_FOVsize//spacing_x)
        target_y = int(xy_FOVsize//spacing_y)

        x_start = int((shape_x - target_x)/2)
        x_last = x_start + target_x

        y_start = int((shape_y - target_y)/2)
        y_last = y_start + target_y

        box_start_, box_end_ = np.array([x_start, y_start, 0]), np.array([x_last, y_last, shape_z])
        return box_start_, box_end_

    def crop_pad(
            self,
            img: torch.Tensor,
            box_start: np.ndarray,
            box_end: np.ndarray,
            mode: str | None = None,
            lazy: bool = False,
            **pad_kwargs,
    ) -> torch.Tensor:
        """
        Crop and pad based on the bounding box.

        """
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices, lazy=lazy)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(
            box_end - np.asarray(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]), 0
        )
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(
            cropped.peek_pending_shape() if isinstance(cropped, MetaTensor) else cropped.shape[1:]
        )
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, lazy=lazy, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta() and isinstance(ret, MetaTensor):
            if not lazy:
                ret.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = ret.applied_operations.pop()
            else:
                pad_info = ret.pending_operations.pop()
                crop_info = ret.pending_operations.pop()
                extra = crop_info[TraceKeys.EXTRA_INFO]
                extra["pad_info"] = pad_info
                self.push_transform(
                    ret,
                    orig_size=crop_info.get(TraceKeys.ORIG_SIZE),
                    sp_size=pad_info[LazyAttr.SHAPE],
                    affine=crop_info[LazyAttr.AFFINE] @ pad_info[LazyAttr.AFFINE],
                    lazy=lazy,
                    extra_info=extra,
                )
        return ret

    def __call__(  # type: ignore[override]
            self, img: torch.Tensor, mode: str | None = None, lazy: bool | None = None, **pad_kwargs
    ) -> torch.Tensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        #box_start, box_end = self.compute_bounding_box(img)
        box_start, box_end = self.compute_bounding_box(img, xy_FOVsize=self.xy_FOVsize)
        lazy_ = self.lazy if lazy is None else lazy
        cropped = self.crop_pad(img, box_start, box_end, mode, lazy=lazy_, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end  # type: ignore[return-value]
        return cropped

class CropForeground_CustomFOVsized(Cropd):
    """
    Dictionary-based version :py:class:`monai.transforms.CropForeground_CustomFOVsize`.
    """

    @deprecated_arg_default("allow_smaller", old_default=True, new_default=False, since="1.2", replaced="1.5")
    def __init__(
            self,
            keys: KeysCollection,
            source_key: str,
            select_fn: Callable = is_positive,
            channel_indices: IndexSelection | None = None,
            margin: Sequence[int] | int = 0,
            allow_smaller: bool = True,
            k_divisible: Sequence[int] | int = 1,
            mode: SequenceStr = PytorchPadMode.CONSTANT,
            start_coord_key: str | None = "foreground_start_coord",
            end_coord_key: str | None = "foreground_end_coord",
            allow_missing_keys: bool = False,
            lazy: bool = False,
            xy_FOVsize = 120,
            **pad_kwargs,
    ) -> None:
        """
        Default, no change from "CropForeground"
        """
        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        self.xy_FOVsize = xy_FOVsize
        cropper = CropForeground_CustomFOVsize(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            lazy=lazy,
            **pad_kwargs,
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys, lazy=lazy)
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    @property
    def requires_current_data(self):
        return True

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropForeground_CustomFOVsize
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key], xy_FOVsize=self.xy_FOVsize)
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d

class SpatialPadSquared(Padd):
    """
    Dictionary-based wrapper of custom function of "SpatialPadSquare".
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """
    def __init__(
            self,
            keys: KeysCollection,
            #spatial_size: Sequence[int] | int,
            method: str = Method.SYMMETRIC,
            mode: SequenceStr = PytorchPadMode.CONSTANT,
            allow_missing_keys: bool = False,
            lazy: bool = False,
            **kwargs,) -> None:

        LazyTransform.__init__(self, lazy)
        padder = SpatialPadSquare(method, lazy=lazy, **kwargs)
        Padd.__init__(self, keys=keys, padder=padder, mode=mode, allow_missing_keys=allow_missing_keys)