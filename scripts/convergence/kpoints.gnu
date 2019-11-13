set term post eps color enh solid 
datafile='toten_vs_kpoints.dat'
set output 'kpoints.eps'
set title 'toten\_vs\_kpoints' font 'Times_New_Roman,24'
set xlabel 'kpoints' font 'Times_New_Roman,18'
set ylabel 'TOTEN' font 'Times_New_Roman,18'
plot datafile  with linespoints lt 7 lw 4 pt 9 pointsize 2
set output
