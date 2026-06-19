# Imports diretos — SEM try/except mascarando o erro real. Se uma dependência
# faltar, o ImportError verdadeiro (ex.: "No module named 'mlflow'") propaga, em
# vez de virar um enganoso "cannot import name ... / No module named 'helper'".
#
# Em contexto de pacote (ex.: api.pedrita.v3) usamos imports relativos; rodando
# como script (`python v3`, sem pacote), usamos imports absolutos.
if __package__:
	from . import helper
	from . import tracking
	from . import context_base
	from . import gemini
	from . import openai_vision
	from .metadata import *
	# from . import localize  # disabled
	from .dset import *
	from .predict import *
	# from .video import *  # disabled
	from .train import *
else:
	import helper
	import tracking
	import context_base
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
