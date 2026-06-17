#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
import __init__ as pedrita
from datetime import datetime as dt


def parse_args(argv) -> argparse.Namespace:
	parser = argparse.ArgumentParser(prog='pedrita')
	sub = parser.add_subparsers(dest='cmd', required=True)

	p_train = sub.add_parser('train', help='train a model')
	p_train.add_argument('--epochs', '-e', type=int, default=2)
	p_train.add_argument('--image', '-i', required=True, help='image, or folder, or nested subfolders train and test, with real and fake subfolders')
	p_train.add_argument('--owarm', '-ow', type=int, default=1, help='number of epochs to train before applying Online Hard Example Mining (OHEM)')
	p_train.add_argument('--limit', '-l', type=int, default=None, help='limit the number of training samples (for quick tests)')
	p_train.add_argument('--freeze', '-z', type=int, default=2, help='number of top-level model children to freeze (negative to count from the end)')
	p_train.add_argument('--oalpha', '-a', type=float, default=0.5, help='weight for OHEM loss when applied (between 0 and 1)')

	p_test = sub.add_parser('test', help='run prediction / evaluation')
	p_test.add_argument('--image', '-i', default=None,
		help='path to test folder (subfolders per label) to compute accuracy')
	p_test.add_argument('--cpu', action='store_true', default=False, help='force CPU usage')
	p_test.add_argument('--limit', '-l', type=int, default=None, help='limit the number of evaluation samples (for quick tests)')

	# 'video' and 'detect' subcommands are disabled (video.py / localize.py commented out).
	return parser.parse_args(argv)

def main():
	argv = list(sys.argv[1:])

	first = argv[0]
	if first == 'merge':
		argv = argv[1:]
		return cli_merge(argv)
	if first == 'gemini':
		return cli_gemini(argv[1:])
	model_path = Path(argv.pop(1))
	if not model_path.is_file():
		raise ValueError(f'Please provide a valid model') from None
	
	argv = parse_args(argv)
	argv.image = Path(argv.image) if argv.image else None
	pedrita.best_device(getattr(argv, 'cpu', False))
	pedrita.set_model(model_path)
	pedrita.online_training(None, None) # type: ignore

	now = dt.now()
	dict(
		train=cli_train,
	  	test=cli_test,
	)[argv.cmd](argv)
	print(dt.now() - now)
def cli_test(argv):
	# If evaluation requested, run and exit
	if argv.image.is_dir():
		pedrita.evaluate_folder(argv.image, limit=argv.limit)
		return
	elif not argv.image.is_file():
		raise ValueError('Please provide a valid image file') from None

	prob, cam_img = pedrita.heatmap(argv.image, minmax=True)
	print(f'Proba real: {prob*100:.2f} %')

	if cam_img is not None:
		fname = 'heatmap.png'
		cam_img.save(fname)
		print(fname)

def cli_train(argv):
	if not argv.image.is_dir():
		raise ValueError('Please provide a folder') from None
	
	pedrita.train(argv.image, 
		epochs=argv.epochs, 
		limit=argv.limit, 
		freeze=argv.freeze, 
		ohwarmup=argv.owarm,
		ohalpha=argv.oalpha,
	)
	test_dir = argv.image / 'test'
	if not test_dir.is_dir(): return

	print('# Evaluate')
	pedrita.evaluate_folder(test_dir, 
		limit=argv.limit*0.2 if argv.limit else None)

def cli_gemini(argv):
	import json as _json
	ap = argparse.ArgumentParser(prog='pedrita gemini', description='Gemini contextual analysis (JSON)')
	ap.add_argument('--image', '-i', required=True, help='path to image file')
	ap.add_argument('--model', '-m', default=None, help='override Gemini model id')
	ap.add_argument('--lang', default='Portuguese', help='language for description/opinion')
	a = ap.parse_args(argv)
	img = Path(a.image)
	if not img.is_file():
		raise ValueError('Please provide a valid image file') from None
	result = pedrita.gemini.context(img, model=a.model, lang=a.lang)
	print(_json.dumps(result, ensure_ascii=False, indent=2))

# cli_detect / cli_video are disabled (localize.py / video.py commented out).
# def cli_detect(argv):
# 	img = argv.image
# 	if not img or not img.is_file():
# 		raise ValueError('Please provide a valid image file') from None
# 	names = [s for s in argv.sources.split(',') if s.strip()]
# 	localizers = pedrita.localize.build(names, grid=argv.grid)
# 	result = pedrita.localize.fuse(img, localizers, alpha=argv.alpha)
# 	for s in result['sources']:
# 		if s['ok']:
# 			print(f"  {s['name']}: score_fake={s['score_fake']*100:.2f} %")
# 		else:
# 			print(f"  {s['name']}: FAILED ({s['error']})")
# 	if result['score_fake'] is not None:
# 		print(f"Fused score_fake: {result['score_fake']*100:.2f} %")
# 	fname = 'detect_heatmap.png'
# 	result['heatmap'].save(fname)
# 	print(fname)
#
# def cli_video(argv):
# 	results = pedrita.predict_video(video_path=argv.video, num_frames=argv.nframes)
# 	print(results)

def cli_merge(argv):
	# model paths in order
	# destination is third path or models/merged.pkl
	mnames = [Path(p) for p in argv[0:2]]
	outname = Path(argv[2]) if len(argv) >= 3 else Path('models/merged.pkl')
	if not all(m.is_file() for m in mnames):
		raise ValueError(f'Please provide two valid model files to merge') from None
	m_a = pedrita.set_model(mnames[0])
	m_b = pedrita.set_model(mnames[1])
	out_model = pedrita.merge(m_a, m_b)
	pedrita.set_model(out_model)
	print('Merged model: ')
	pedrita.save_model(outname)
	
if __name__ == '__main__': main()
