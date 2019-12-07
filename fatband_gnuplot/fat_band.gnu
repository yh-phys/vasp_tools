set encoding iso_8859_1
# set terminal  postscript enhanced color font "TimesNewRoman, 11" size 5, 4
set terminal  pngcairo truecolor enhanced lw 5.0 font "TimesNewRoman, 44" size 1920, 1680
set palette rgbformulae 22, 13, -31
# set palette rgbformulae 7,5,15
set output 'Se_s.png'
set border
unset colorbox
set title "Se\\_s" offset 0, -0.8 font "TimesNewRoman, 54"
set style data linespoints
unset ztics
unset key
# set key outside top vertical center
# set pointsize 0.3
set view 0,0
set xtics font "TimesNewRoman, 44"
set xtics offset 0, 0.3
set ytics font "TimesNewRoman, 40"
set ytics -4, 2, 4
set ylabel font "TimesNewRoman, 48"
set ylabel offset 1.0, 0
set xrange [0:2.426]
set ylabel "Energy (eV)"
set yrange [-4:4]
set xtics ("M" 0.00000, "G" 0.888, "K" 1.913, "M" 2.426)
plot -4 with filledcurves y1=4 lc rgb "navy", \
'PBAND_Se.dat' u ($1):($2):($3) w lines lw 1.5 lc palette, \
0 w l dt 2 lc rgb "gray", \
'< echo "0.888 -4 \n 0.888 4"' w l dt 2 lc rgb "gray", \
'< echo "1.913 -4 \n 1.913 4"' w l dt 2 lc rgb "gray"
