for i in {1..3}
do  
    declare -i n
    n=$i-1
    sed -i 's/crossover_type '"$n"'/crossover_type '"$i"'/' constants_serial.h
    sed -i 's/crossover_type '"$n"'/crossover_type '"$i"'/' constants.h
    make_hybrid="Hybrid_Model"
    make_global="Parallel_GA_Island_Global_Mem"
    make_shared="Parallel_GA_Island_Shared_Mem"
    make_ms="Parallel_Main_Secondary_Sphere_Optimization_GA"
    make_serial="Serial_Sphere_Optimization_GA"

    make clean -C $make_hybrid
    make -C $make_hybrid

    make clean -C $make_global
    make -C $make_global

    make clean -C $make_shared
    make -C $make_shared

    make clean -C $make_ms
    make -C $make_ms

    make clean -C $make_serial
    make -C $make_serial

    output=${make_hybrid##*/}
    output1=${make_global##*/}
    output2=${make_shared##*/}
    output3=${make_ms##*/}
    output4=${make_serial##*/}

    for i in {0..9}
    do
        # Running hybrid GA
        (/usr/bin/time -f "%e,%U,%S" ./$make_hybrid/ga) 2>> ${output%.*}.txt

        # Running global mem GA
        (/usr/bin/time -f "%e,%U,%S" ./$make_global/ga) 2>> ${output1%.*}.txt

        # Running shared mem GA
        # (/usr/bin/time -f "%e,%U,%S" ./$make_shared/ga) 2>> ${output2%.*}.txt

        # Running main secondary GA
        (/usr/bin/time -f "%e,%U,%S" ./$make_ms/ga) 2>> ${output3%.*}.txt

        # Running serial GA
        (/usr/bin/time -f "%e,%U,%S" ./$make_serial/ga) 2>> ${output4%.*}.txt

    done
done



