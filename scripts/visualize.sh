export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/Margin_Stage2_Full.py \
	--result-path work_dirs/Margin_Stage2_Full/results.pkl\
	--out-dir vis_long_Margin_Stage2_Full