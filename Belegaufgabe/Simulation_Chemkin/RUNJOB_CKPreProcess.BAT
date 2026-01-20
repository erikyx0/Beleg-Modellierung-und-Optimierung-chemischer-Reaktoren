REM Generated Batch Job Command File

REM get the runtime environment defined

CALL "C:\Program Files\ANSYS Inc\v231\reaction\chemkinpro.win64\bin\run_chemkinpro_env_setup.bat"
REM cd to the Working Directory for this job
\\ && cd zfs1.hrz.tu-freiberg.de\ed67qowe\home\fak1pool\Desktop\Repos\Belegaufgabe\Simulation_Chemkin
SET MKL_NUM_THREADS=1
SET OMP_NUM_THREADS=1
SET ANSYS_PARENT_CHILD_ID=dc0df5e3-cab5-4e96-baa9-eac74d719496
SET ANSYS_PARENT_CHILD_SERVER=53643@R28.adds.tu-freiberg.de
SET ANSYS_LICENSING_CONTEXT_DATA="<ANSYS_LICENSING_CONTEXT_DATA><ID>dc0df5e3-cab5-4e96-baa9-eac74d719496</ID><TYPE>PARENT_CHILD</TYPE><NAME>Parent Child</NAME><HANDLER>53643@R28.adds.tu-freiberg.de</HANDLER></ANSYS_LICENSING_CONTEXT_DATA>"
SET XERCES_DISABLE_DTD=1

REM RUN THE JOB AND CAPTURE EXIT STATUS

CKPreProcess.EXE -i "\\zfs1.hrz.tu-freiberg.de\ed67qowe\home\fak1pool\Desktop\Repos\Belegaufgabe\Simulation_Chemkin\CKPreProc_Mechanismus_PT_CH4.input"
exit %errorlevel%
