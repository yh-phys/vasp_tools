set term post eps color enh solid 
datafile='TDOS.dat'
set output 'TDOS.eps'
set xlabel 'Energy (eV)' font 'Times_New_Roman,18'
set ylabel 'TDOS (states/ev)' font 'Times_New_Roman,18'
set xrange [-5:5]
set yrange [0:15]
set bmargin 4
set style fill transparent solid 0.4
plot datafile title "TDOS" with filledcurves y1=0 lw 2 lc rgb "dark-gray"
set output
