from typing import Union, Optional
from tokenizers import Tokenizer
from tokenizers.models import Model
from tokenizers.trainers import Trainer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.normalizers import Normalizer
from torch.utils.data import Dataset, Sampler, DataLoader
from time import time
import os
from warnings import warn


def build_tokenizer(
    model: Union[str, Model],
    trainer: Optional[Trainer] = None,
    train_dataset: Optional[Dataset] = None,
    train_batch_sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    save_to_json_file: Optional[str] = None,
    override_normalizer: Optional[Normalizer] = None,
    override_pre_tokenizer: Optional[PreTokenizer] = None,
    override_post_processor: Optional[PostProcessor] = None,
    full_cycles_num: Optional[int] = None,
    iterations_num: Optional[int] = None,
    time_limit_minutes: Optional[int] = None,
    stop_filename: Optional[str] = None,
) -> Tokenizer:
    """
    Helper function to create tokenizers, used for both protein and small-molecules tokenization
    Args:
        model: either a str, in this case 'iupac' or 'unirep' are accepted.
            alternatively you can provide a tokenizers.Model instance - for example BPE or WordLevel
        save_to_json_file: optional. If provided, a str is expected with a path to the output json file which can be used to load the tokenizer.

    """
    if (
        full_cycles_num is None
        and iterations_num is None
        and time_limit_minutes is None
    ):
        full_cycles_num = 1

    if (trainer is not None) ^ (train_dataset is not None):
        raise Exception(
            'You provided only "trainer" or "train_on_data", you must provide neither or both!'
        )

    assert isinstance(model, Model)
    tokenizer = Tokenizer(model=model)

    #### normalize

    # NOTE: not doing any normalization because they don't natively support Uppercasing,
    #      and if I add a custom normalizer the tokenizer can't save/serialize ...
    #      instead, we'll have it in fuse data/sample pipeline
    if override_normalizer is not None:
        tokenizer.normalizer = override_normalizer

    #### pre_tokenize
    if override_pre_tokenizer is not None:
        tokenizer.pre_tokenizer = override_pre_tokenizer

    ####post_process
    if override_post_processor is not None:
        tokenizer.post_processor = override_post_processor
    else:
        warn(
            "no post_processor provided. This is usually ok if you are 1. Creating a tokenizer in order to store it in a file, as changing post-processor does not require retraining the tokenizer. 2. Intend to override the post-processor in the returned tokenizer."
        )

    ######encoded = tokenizer.encode('a  AbCC ddd     \n\r\n\t \t Zz')

    # encoding a pair:
    # tokenizer.encode('a  AbCC ddd     \n\r\n\t \t Zz', 'bAnAnA') #.sequence_ids
    # encoding batch of pairs:
    # tokenizer.encode_batch([('a  AbCC ddd     \n\r\n\t \t Zz', 'bAnAnA'),('how are', 'cocolala')]) #[1].token

    # encoded.tokens
    # encoded.ids
    # encoded.sequence_ids

    if trainer is not None:

        tokenizer.train_from_iterator(
            iterator_func(
                train_dataset=train_dataset,
                train_batch_sampler=train_batch_sampler,
                full_cycles_num=full_cycles_num,
                iterations_num=iterations_num,
                time_limit_minutes=time_limit_minutes,
                stop_filename=stop_filename,
                num_workers=num_workers,
            ),
            trainer=trainer,
            length=len(train_dataset),
        )

    if save_to_json_file is not None:
        tokenizer.save(save_to_json_file)

    return tokenizer


def iterator_func(
    train_dataset: Dataset,
    train_batch_sampler: Sampler,
    full_cycles_num: int,
    iterations_num: int,
    time_limit_minutes: int,
    stop_filename: str,
    num_workers: int = 0,
) -> None:
    # TODO: allow multiprocessing (but consider collate and pickles/copies...)
    dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
    )

    is_work_done = IsWorkDone(
        full_cycles_num, iterations_num, time_limit_minutes, stop_filename
    )
    curr_cycle = -1
    curr_iter = -1
    should_stop = False

    while True:
        curr_cycle += 1
        for curr in iter(dataloader):
            curr_iter += 1
            if not curr_iter % 100_000:
                print("iterator_func:curr_iter=", curr_iter)
            if is_work_done.update(curr_cycle, curr_iter):
                print("DONE!")
                should_stop = True
                break
            if curr is None:
                continue
            if isinstance(curr, list):
                for curr_str in curr:
                    if not isinstance(curr_str, str):
                        raise Exception("unexpected type!")
                    if 0 == curr_iter:
                        print(
                            f"printing sample information because it is the first iter minibatch: {curr_str}"
                        )
                    yield curr_str
            else:
                raise Exception("expected minibatch to be a list")

        if should_stop:
            break

    return


class IsWorkDone:
    def __init__(
        self,
        full_cycles_num: Optional[int] = None,
        iterations_num: Optional[int] = None,
        time_limit_minutes: Optional[int] = None,
        stop_filename: str = None,
    ):
        if full_cycles_num is not None:
            full_cycles_num = int(full_cycles_num)

        if iterations_num is not None:
            iterations_num = int(iterations_num)

        if time_limit_minutes is not None:
            time_limit_minutes = float(time_limit_minutes)

        self.mode_selected = None
        error_msg = "only one of full_cycles_num, iterations_num, time_limit_minutes is allowed. And you must choose one."
        self.start_time = time()
        if iterations_num is not None:
            if self.mode_selected is not None:
                raise Exception(error_msg)
            self.mode_selected = "iterations"
            print(f"iterator_func: selected mode: {iterations_num} iterations")
            self.curr_iter = 0
            self.iterations_num = iterations_num
        elif time_limit_minutes is not None:
            if self.mode_selected is not None:
                raise Exception(error_msg)
            self.mode_selected = "time_limit"
            print(
                f"iterator_func: selected mode: {time_limit_minutes} minutes time_limit"
            )
            self.time_limit_minutes = time_limit_minutes

        elif full_cycles_num is not None:
            if self.mode_selected is not None:
                raise Exception(error_msg)
            self.mode_selected = "full_cycles"
            print(f"iterator_func: selected mode: {full_cycles_num} full_cycles")
            self.curr_cycle = 0
            self.full_cycles_num = full_cycles_num
        else:
            raise Exception(
                "IsWorkDone::No mode selected! you must choose one of full_cycles_num, iterations_num or time_limit_minutes"
            )

        if self.mode_selected is None:
            raise Exception(error_msg)

        self.stop_filename = stop_filename
        if self.stop_filename is not None:
            print("stop_filename for fast tokenizer learn is ", stop_filename)

        self._last_printed_hour = -1
        self._last_printed_epoch = -1

    def update(self, epoch_num: int, iteration_num: int) -> bool:
        """
        if returns True it means we need to stop
        """
        minutes_passed = (time() - self.start_time) / 60.0
        hours_passed = int(minutes_passed) // 60
        if hours_passed > self._last_printed_hour:
            if hasattr(self, "time_limit_minutes"):
                print(f"{hours_passed} hours of total {self.time_limit_minutes/60:.2f}")
            else:
                print(f"{hours_passed} hours so far")
            self._last_printed_hour = hours_passed

        if epoch_num > self._last_printed_epoch:
            print(f"training tokenizer - epoch {epoch_num}")
            self._last_printed_epoch = epoch_num

        if self.mode_selected == "full_cycles":
            if epoch_num >= self.full_cycles_num:
                print(
                    f"reached end of {epoch_num} full cycles, after {iteration_num} iterations"
                )
                return True
        elif self.mode_selected == "iterations":
            if iteration_num >= self.iterations_num:
                print(f"reached end of {self.iterations_num} iterations")
                return True
        elif self.mode_selected == "time_limit":
            if minutes_passed >= self.time_limit_minutes:
                print(
                    f"reached end of {self.time_limit_minutes} minutes, after {iteration_num} iterations"
                )
                return True
        if (self.stop_filename is not None) and (not iteration_num % 1000):
            if os.path.isfile(self.stop_filename):
                print(
                    f'stop filename = "{self.stop_filename}" encountered! stopping tokenizer training. After {iteration_num} iterations'
                )
                return True

        return False


# TODO: add usage example/test
