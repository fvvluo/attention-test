# Alias module so that `from flash_attn_3.flash_attn_interface import ...` works.
from flash_attn_interface import *  # noqa: F401,F403
from flash_attn_interface import (  # noqa: F401
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    flash_attn_combine,
    get_scheduler_metadata,
)
