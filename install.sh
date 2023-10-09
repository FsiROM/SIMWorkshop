pyth_pip_path=$(python -c "exec(\"import site\nprint(site.getsitepackages()[0])\")")


pip install KratosMultiphysics==9.4
pip install KratosLinearSolversApplication==9.4 > /dev/null 2>&1
pip install KratosFluidDynamicsApplication==9.4 > /dev/null 2>&1
pip install KratosStructuralMechanicsApplication==9.4 > /dev/null 2>&1
pip install KratosMeshMovingApplication==9.4 > /dev/null 2>&1
pip install KratosMappingApplication==9.4 > /dev/null 2>&1
pip install KratosConstitutiveLawsApplication==9.4 > /dev/null 2>&1
pip install KratosCoSimulationApplication==9.4 > /dev/null 2>&1

rm -r $pyth_pip_path/KratosMultiphysics/CoSimulationApplication/
cp -r FsiROM_Kratos_patch/CoSimulationApplication/ $pyth_pip_path/KratosMultiphysics/CoSimulationApplication/
cp FsiROM_Kratos_patch/fluid_solver.py $pyth_pip_path/KratosMultiphysics/FluidDynamicsApplication/


git clone --single-branch --branch Workshop https://github.com/azzeddinetiba/ROM_AM.git > /dev/null 2>&1
pip install ROM_AM/.
wget "https://fem-on-colab.github.io/releases/vtk-install.sh" -O "/tmp/vtk-install.sh" && bash "/tmp/vtk-install.sh" > /dev/null 2>&1

export OMP_NUM_THREADS=1

