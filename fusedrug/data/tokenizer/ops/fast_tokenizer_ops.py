from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
from tokenizers import Tokenizer
from warnings import warn
from collections import defaultdict

class FastTokenizer(OpBase):
    '''
    applies a tokenizers (https://github.com/huggingface/tokenizers) based tokenizer
    '''

    def __init__(self, 
        tokenizer_json, 
        max_size=None, #determines both padding length and max size.
        pad_id=None, 
        pad_type_id=None,        
        verbose=0,
        **kwargs
        ):
        super().__init__(**kwargs)

        if verbose>0:
            print(f'DEBUG:FastTokenizer __init__ called for json {tokenizer_json}')
        
        self._tokenizer_json = tokenizer_json
        self._tokenizer = Tokenizer.from_file(self._tokenizer_json)      
        self._pad_id = pad_id  
        self._verbose = verbose        

        if max_size is not None:
            assert isinstance(max_size, int)
            assert max_size>0
            assert isinstance(pad_id, int)

            padding_kwargs = dict(length=max_size, pad_id=pad_id)
            if pad_type_id is not None:
                assert isinstance(pad_type_id, int)
                padding_kwargs['pad_type_id'] = pad_type_id

            self._tokenizer.enable_padding(
                direction='right',
                **padding_kwargs
            )

            self._tokenizer.enable_truncation(
                max_length=max_size,
                direction='right',
            )

        self._max_size = max_size

        if self._verbose>0:
            self._debug_max_tokenized_len_encountered = defaultdict(int)
        
    #note: use normalizer.Sequence to chain multiple normalizers
    def set_normalizer(self, normalizer):
        self._tokenizer.normalizer = normalizer

    #note: use pre_toknizers.Sequence to chain multiple pre_toknizers
    def set_pre_tokenizer(self, pre_tokenizer):
        self._tokenizer.pre_tokenizer = pre_tokenizer

    def set_post_processor(self, post_processor):
        self._tokenizer.post_processor = post_processor
        
    def get_vocab_size(self):
        return self._tokenizer.get_vocab_size()

    def get_token_id(self, token_str):
        ans = self._tokenizer.token_to_id(token_str)
        assert ans is not None, f'could not find token id for token:{token_str}!'
        return ans

    def __call__(self, sample_dict: NDict, key_in, 
        key_out_tokenized_object:str=None, 
        key_out_tokens_ids:str=None, 
        key_out_attention_mask:str=None, 
        convert_attention_mask_to_bool=True,
        ):
        data_str = sample_dict[key_in]
        if not isinstance(data_str, str):            
            raise Exception(f'Expected key_in={key_in} to point to a string, and instead got a {type(data_str)}. value={data_str}')

        encoded = self._tokenizer.encode(data_str)

        if self._max_size is not None: #we tightly couple padding length and max size.
            assert self._max_size == len(encoded.ids)

        if self._verbose>0:
            if self._pad_id in encoded.ids:
                _encoded_len_unpadded = encoded.ids.index(self._pad_id)
            else:
                #no padding, therefore it was fully used (either exactly the size, or most likely it was clipped)
                _encoded_len_unpadded = len(encoded.ids)

            if _encoded_len_unpadded > self._debug_max_tokenized_len_encountered[self._tokenizer_json]:
                print('DEBUG: FastTokenizer: encountered new max encoded size:', _encoded_len_unpadded, ' for tokenizer: ', self._tokenizer_json)
                self._debug_max_tokenized_len_encountered[self._tokenizer_json] = _encoded_len_unpadded
       
        #KEEP THIS AS DOC FOR NOW
        #encoded has attributes [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]
        # ids are the encoded tokens, 
        # type_ids are for things like "which sentence is this from"
        # tokens are the actual tokens (for example - ['c1ccc(', '/C(', '=N/N', 'c2nc3ccccc3', 's2)', 'c2cccc', 'n2)cc1', '[PAD]', '[PAD]', '[PAD]'])
        # offsets describe the starting point and length of each original token
        # attention_mask - by default puts 1 for everything that isn't padding, and 0 for those that are padding
        # special_tokens_mask - 1 for anything that is a special token (e.g. padding, separator, etc.) 0 for the rest
        # overflowing - I *assume* it's any original content that get clipped out due to max length definition

        if len(encoded.overflowing)>0:
            print(f'Warning: FastTokenizer had to truncate sequence. Original Sequence Length = {len(data_str)} max supported = {self._max_size} for tokenizer: {self._tokenizer_json} for sample_id {get_sample_id(sample_dict)}')

        if key_out_tokenized_object is not None:
            #if requested, store the entire tokenizer.Encoding object (which provides access to attributes such as  [ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
            sample_dict[key_out_tokenized_object] = encoded

        if key_out_tokens_ids is not None:
            sample_dict[key_out_tokens_ids] = encoded.ids        

        if key_out_attention_mask is not None:
            sample_dict[key_out_attention_mask] = encoded.attention_mask   
            if convert_attention_mask_to_bool:     
                sample_dict[key_out_attention_mask] = [bool(x) for x in sample_dict[key_out_attention_mask]]

        if (key_out_tokens_ids is None) and (key_out_tokenized_object is None):
            warn('FastTokenizer Op got key_out_tokens_ids=None and key_out_tokenized_object=None, which means it will not modify anything in the sample. Is this intended?')
        
        return sample_dict