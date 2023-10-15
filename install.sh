pythcmd="python"
while getopts p: flag
do
    case "${flag}" in
        p) pythcmd=${OPTARG};;
    esac

done
pyth_pip_path=$($pythcmd -c "exec(\"import site\nprint(site.getsitepackages()[0])\")")

printf "\n============= Installing KratosMultiphysics and necessary applications\n"
pip install KratosMultiphysics==9.4 > /dev/null 2>&1
pip install KratosLinearSolversApplication==9.4 > /dev/null 2>&1
pip install KratosFluidDynamicsApplication==9.4 > /dev/null 2>&1
pip install KratosStructuralMechanicsApplication==9.4 > /dev/null 2>&1
pip install KratosMeshMovingApplication==9.4 > /dev/null 2>&1
pip install KratosMappingApplication==9.4 > /dev/null 2>&1
pip install KratosConstitutiveLawsApplication==9.4 > /dev/null 2>&1
pip install KratosCoSimulationApplication==9.4 > /dev/null 2>&1

printf "\n============= Installing KratosMultiphysics extension for ROM-FOM simulations (See https://github.com/FsiROM/Kratos )\n"
rm -r $pyth_pip_path/KratosMultiphysics/CoSimulationApplication/
cp -r FsiROM_Kratos_patch/CoSimulationApplication/ $pyth_pip_path/KratosMultiphysics/CoSimulationApplication/
cp FsiROM_Kratos_patch/fluid_solver.py $pyth_pip_path/KratosMultiphysics/FluidDynamicsApplication/

printf "\n============= Installing ROM_AM package for non intrusive model reduction (See https://github.com/azzeddinetiba/ROM_AM )\n"
git clone --single-branch --branch Workshop https://github.com/azzeddinetiba/ROM_AM.git > /dev/null 2>&1
pip install ROM_AM/. > /dev/null 2>&1


export OMP_NUM_THREADS=1
