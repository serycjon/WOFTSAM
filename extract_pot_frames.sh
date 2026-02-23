src_dir=POT-210/orig_videos
dst_dir=POT-210/extracted/images
mkdir -p $dst_dir
for obj_id in $(seq -w 30); do
    for seq_type in $(seq 7); do
        echo "extracting $seq_id"
        seq_id="V${obj_id}_${seq_type}"
        seq_dir="${dst_dir}/${seq_id}"
        mkdir -p "$seq_dir"
        cd "$seq_dir"
        src_vid="${src_dir}/V${obj_id}/${seq_id}.avi"
        ffmpeg -i "$src_vid" -q:v 1 -vsync 0 -f image2 %08d.jpg
    done
done
