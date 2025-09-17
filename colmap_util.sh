#!/usr/bin/env bash
EXT='tif'
FOLDER_IMG=images

:<<END
DIR_COLMAP=/mnt/hdd_16tb/dataset_stereo/dabeeo/samsung_dong
#DIR_FILTERED=/mnt/hdd_16tb/dataset_stereo/dabeeo/samsung_dong_mini_51/
#ENDS_IMG=( "800774 800781" "800884 800891" "800961 800969" "801069 801077" "801139 801147" "801241 801250" "801302 801309")
END

:<<END
DIR_COLMAP=/mnt/hdd_16tb/dataset_stereo/dabeeo/samsung_dong
DIR_FILTERED=/mnt/hdd_16tb/dataset_stereo/dabeeo/samsung_dong_mini_30/
ENDS_IMG=( "800885 800890" "800962 800967" "801071 801076" "801140 801145" "801244 801249" )
END

ENDS_IMG=( "00020 00026" "100022 100027" "100110 100115" "100162 100164" "1200042 1200047" "1200090 1200091" )
DIR_COLMAP=/media2/data/dataset_stereo/sillim_ew/


LIST_IMG=$(
    for pair in "${ENDS_IMG[@]}"; do
    # pair 변수는 e.g. "800776 800778"
        set -- $pair             # 인자를 분리하면 $1=800776, $2=800778
        # Get the length of the first string to preserve zero-padding
        len=${#1}
        # Force base-10 interpretation to avoid octal issues with leading zeros
        start=$((10#$1))
        end=$((10#$2))
        for ((i = start; i <= end; i++)); do
            printf "%0${len}d.%s " "$i" "$EXT"
        done
    done
)

:<<END
# Use the new list mode (no output directory needed, auto-generated)
echo "LIST_IMG : $LIST_IMG" #&& exit
#python3 colmap_util.py list $DIR_COLMAP $FOLDER_IMG "$LIST_IMG"
END

# Example of using dist mode (no output directory needed, auto-generated)
# To select 20 closest images to geometric median:
python3 colmap_util.py dist $DIR_COLMAP $FOLDER_IMG 20

# To select 10% of images closest to geometric median:
# python3 colmap_util.py dist $DIR_COLMAP $FOLDER_IMG 0.1
