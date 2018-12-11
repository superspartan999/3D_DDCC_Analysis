shopt - s extglob
chmod 777 *
gmsh - 3 p_structure_0.17_10nm.geo
ifort PBC30.f90
./a.out p_structure_0.17_10nm.msh
./compile.bat
./p_structure_0.17_10nm _Copy.sh
./p_structure_0.17_10nm _Jobs.sh
