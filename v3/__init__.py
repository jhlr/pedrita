try:
	# Import submodules as modules to keep live references (avoid `from ... import *`)
	from . import helper
	from . import tracking
	from . import gemini
	from . import openai_vision
	from .metadata import *
	# from . import localize  # disabled
	from .dset import *
	from .predict import *
	# from .video import *  # disabled
	from .train import *
except Exception:
	# Fallback for environments where package context still isn't available
	import helper
	import tracking
	import gemini
	import openai_vision
	from metadata import *
	# import localize  # disabled
	from dset import *
	from predict import *
	# from video import *  # disabled
	from train import *

# Convenience re-exports for commonly used helpers
set_model = helper.set_model
best_device = helper.best_device

def __getattr__(name: str): return getattr(helper, name)


