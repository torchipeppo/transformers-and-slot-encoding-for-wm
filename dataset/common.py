import torch
import einops
from dataclasses import dataclass



# pads by repeating the final frame of the video, also returns padding mask
# padding mask: True is for frames that are to be ignored (like torch attention layer)
def make_batch_padded(list_of_samples):
    max_len = 0
    for video in list_of_samples:
        t = video.shape[0]
        if t > max_len:
            max_len = t
    padded_samples = []
    padding_masks = []
    for video in list_of_samples:
        padding = einops.repeat(video[-1, :, :, :], "c w h -> repeat c w h", repeat = max_len-video.shape[0])
        # padded_video = einops.rearrange([video, padding], "listaxis t c w h -> (listaxis t) c w h")
        padded_video = torch.cat([video, padding], dim=0)
        padded_samples.append(padded_video)
        mask = [i>=video.shape[0] for i in range(max_len)]
        mask = torch.Tensor(mask).bool()
        padding_masks.append(mask)
    # listaxis is batch axis
    padded_samples = einops.rearrange(padded_samples, "listaxis t c w h -> listaxis t c w h")
    padding_masks = einops.rearrange(padding_masks, "listaxis m -> listaxis m")
    return padded_samples, padding_masks



def get_min_len(list_of_samples):
    min_len = 999
    for video in list_of_samples:
        t = video.shape[0]
        if t < min_len:
            min_len = t
    return min_len

def make_batch_truncated(list_of_samples, trunc_len=None):
    if trunc_len is None:
        trunc_len = get_min_len(list_of_samples)
    truncated_samples = []
    for video in list_of_samples:
        truncated_samples.append(video[:trunc_len])
    # listaxis is batch axis
    truncated_samples = einops.rearrange(truncated_samples, "listaxis t c w h -> listaxis t c w h")
    return truncated_samples, torch.zeros_like(truncated_samples, dtype=torch.bool)

def make_batch_truncated_Ncurry(n):
    def f(list_of_samples):
        return make_batch_truncated(list_of_samples, n)
    return f



@dataclass
class Batch():
    input_image: torch.Tensor
    target_image: torch.Tensor
    input_padding_mask: torch.Tensor
    target_padding_mask: torch.Tensor
    durations: torch.Tensor
    labels: torch.Tensor = None

    # torch's stance is unclear, but this implementation ALWAYS SIDE-EFFECTS the batch.
    def to(self, *args, **kwargs):
        self.input_image = self.input_image.to(*args, **kwargs)
        self.target_image = self.target_image.to(*args, **kwargs)
        self.input_padding_mask = self.input_padding_mask.to(*args, **kwargs)
        self.target_padding_mask = self.target_padding_mask.to(*args, **kwargs)
        self.durations = self.durations.to(*args, **kwargs)
        if self.labels is not None:
            self.labels = self.labels.to(*args, **kwargs)
        return self

    # image_tensor can be self's own input_image, target_image,
    # or something derived, such as reconstructed logits or images, as long as the first two dimensions match
    # in particular, image_tensor should be missing either the first or the last frame
    def get_last_frames(self, image_tensor):
        ret = torch.stack([ image_tensor[b_index, self.durations[b_index]-1] for b_index in range(self.durations.shape[0]) ], dim=0)
        return ret



def collate_with_future_targets(list_of_samples, make_batch_fn=make_batch_padded):
    # video shape is [t c w h]   (or h w? doesn't matter here anyway)

    # eliminate last frame of the video from the input, b/c we wanna predict the NEXT frame
    # so we don't want to try and predict what comes after the final frame at training time
    # because we'd have no ground truth to compute the loss against
    input_samples = [video[:-1, :, :, :] for video in list_of_samples]
    # shift target vector 1 frame to the left, b/c we're gonna predict the NEXT frame
    target_samples = [video[1:, :, :, :] for video in list_of_samples]
    # -1 b/c we are taking one frame off
    durations = [video.shape[0]-1 for video in list_of_samples]

    input_samples, input_padding_mask = make_batch_fn(input_samples)
    target_samples, target_padding_mask = make_batch_fn(target_samples)
    durations = torch.Tensor(durations).to(int)
    return Batch(input_samples, target_samples, input_padding_mask, target_padding_mask, durations)

# for STEVE comparison
def collate_with_same_targets(list_of_samples, make_batch_fn=make_batch_padded):
    batched_samples, padding_mask = make_batch_fn(list_of_samples)
    durations = torch.Tensor([video.shape[0] for video in list_of_samples]).to(int)
    return Batch(batched_samples, batched_samples, padding_mask, padding_mask, durations)


def collate_fn_factory(collate_fn, make_batch_fn, given_frames=None, labeled=False):

    def batch_videos_only(list_of_samples):
        batch = collate_fn(list_of_samples, make_batch_fn)
        if given_frames is not None:
            batch.input_image = batch.input_image[:, :given_frames]
            batch.input_padding_mask = batch.input_padding_mask[:, :given_frames]
            # do not update durations on purpose, those are the "true" lengths, which are still useful for the targets.
        return batch

    def batch_labeled_videos(list_of_tuples):
        list_of_videos, list_of_labels = tuple(zip(*list_of_tuples))
        the_batch = batch_videos_only(list_of_videos)
        the_batch.labels = torch.stack(list_of_labels)
        return the_batch

    if labeled:
        return batch_labeled_videos
    else:
        return batch_videos_only
