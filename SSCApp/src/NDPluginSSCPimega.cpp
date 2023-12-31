/*
 * This software is Copyright by the Board of Trustees of Michigan
 * State University (c) Copyright 2016.
 *
 * Authors: Marco A. B Montevechi Filho, Henrique F. Simoes, Eduardo X. S. Miqueles
 */
#include <algorithm>
#include <limits>

#include <fstream>

#include <string.h>

#include <epicsMath.h>

#include <iocsh.h>
#include <epicsExport.h>

#include <NDPluginSSCPimega.h>

static const char *driverName = "NDSSCPimega";

asynStatus NDPluginSSCPimega::loadMatrix(int load){

    if(load!=1){ //Only process on record rising from 0 to 1
        return asynSuccess;
    }

    getIntegerParam(blockSize,           &blockSizeVal);
    getIntegerParam(pimegaModel,         &pimegaModelVal);
    getIntegerParam(validWorkspace,      &validWorkspaceVal);
    getStringParam(matrixFilePath,  256, matrixFileName);

    char xFilePath[264];
    char yFilePath[264];

    strcpy(xFilePath, matrixFileName);
    strcpy(yFilePath, matrixFileName);

    strcat(xFilePath,"/x540D.b"); //hardcoded for now. Should be a PV.
    strcat(yFilePath,"/y540D.b");

    std::ifstream infilex(xFilePath);
    std::ifstream infiley(yFilePath);

    if(!bool(infilex.good()) || !bool(infiley.good())){
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                  "%s:: Matrix File does not exist.\n",
                  this->portName);
        return asynError;
    };

    int *ix, *iy;

    iy = (int *)malloc( modelPixels[pimegaModelVal] * modelPixels[pimegaModelVal] * sizeof(int) );
    ix = (int *)malloc( modelPixels[pimegaModelVal] * modelPixels[pimegaModelVal] * sizeof(int) );

    FILE *fpx = fopen( xFilePath, "rb");
    FILE *fpy = fopen( yFilePath, "rb");

    if (fpx == NULL || fpy == NULL){
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                  "%s:: Cant open matrix files.\n",
                  this->portName);
        return asynError;
    }
    
    fread(ix, sizeof(int), modelPixels[pimegaModelVal] * modelPixels[pimegaModelVal], fpx);
    fread(iy, sizeof(int), modelPixels[pimegaModelVal] * modelPixels[pimegaModelVal], fpy);

    if ( ferror(ix) || ferror(iy) ){
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                  "%s:: Cant read matrix files.\n",
                  this->portName);
        return asynError;
    }

    if (validWorkspaceVal){
        ssc_pimega_backend_free_plan( &workspace );
    }

    ssc_pimega_backend_create_plan( &workspace, blockSizeVal, pimegaModelVal );
    ssc_pimega_backend_set_plan( &workspace, ix, iy);

    setIntegerParam(validWorkspace, 1);
    callParamCallbacks(0);

    fclose( fpx );
    fclose( fpy );
    free(iy);
    free(ix);

    return asynSuccess;

}

NDPluginSSCPimega::NDPluginSSCPimega(const char *portName, int queueSize, int blockingCallbacks,
             const char *NDArrayPort, int NDArrayAddr,
             int maxBuffers, size_t maxMemory,
             int priority, int stackSize)
    : NDPluginDriver(portName,
                     queueSize,
                     blockingCallbacks,
                     NDArrayPort,
                     NDArrayAddr,
                     1, // maxAddr
                     maxBuffers,
                     maxMemory,
                     asynInt32ArrayMask | asynFloat64ArrayMask | asynOctetMask | asynGenericPointerMask,
                     asynInt32ArrayMask | asynFloat64ArrayMask | asynOctetMask | asynGenericPointerMask,
                     0, // asynFlags
                     1, // autoConnect
                     priority,
                     stackSize,
                     1) // maxThreads
{
    lastinfo.nElements = (size_t)-1; // spoil

    createParam(SSCPimegaBlockSizeString,  asynParamInt32, &blockSize);
    createParam(SSCPimegaModelString,      asynParamInt32, &pimegaModel);
    createParam(SSCPimegaLoadMatrixString, asynParamInt32, &loadMatrixes);
    createParam(SSCPimegaValidWorkspace,   asynParamInt32, &validWorkspace);
    createParam(SSCPimegaFilePath,         asynParamOctet, &matrixFilePath);

    setStringParam(NDPluginDriverPluginType, "NDPluginSSCPimega");
    setStringParam(matrixFilePath,           "/tmp/");

    setIntegerParam(blockSize, 1);

    callParamCallbacks();

}

NDPluginSSCPimega::~NDPluginSSCPimega() {
    assert(false); // never called (asyn ports can't be destory'd)
}

void
NDPluginSSCPimega::processCallbacks(NDArray *pArray)
{

    // pArray is borrowed reference.  Caller will release()
    NDArrayInfo info;
    NDArray *pOutput;
    NDArray *intermediateIn;

    NDPluginDriver::beginProcessCallbacks(pArray);

    getIntegerParam(pimegaModel,         &pimegaModelVal);
    
    size_t dims[2] = {pArray->dims[0].size, pArray->dims[1].size};

    (void)pArray->getInfo(&info);

    if(modelPixels[pimegaModelVal]!=dims[0]){
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
          "%s:: Array size is not coherent with PiMega model.\n",
          this->portName);
          return;
    }

    if(pArray->ndims!=2 || info.xSize==0 || info.ySize==0) {
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                  "%s:: 2D non-empty expected",
                  this->portName);
        return;
    }
    switch(pArray->dataType) {
    case NDInt8:
    case NDUInt8:
    case NDInt16:
    case NDUInt16:
    case NDInt32:
    case NDUInt32:
    case NDFloat32:
    case NDFloat64:
        break;
    default:
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                  "%s:: Unsupported type %u",
                  this->portName, (unsigned)pArray->dataType);
        return;
    }
    asynPrint(pasynUserSelf, ASYN_TRACE_FLOW, "%s: %s ndarray=%p\n", this->portName, __PRETTY_FUNCTION__, pArray);

    pOutput = this->pNDArrayPool->alloc(2, dims, NDFloat32, 0, NULL);

    if (pArray->dataType != NDFloat32) {
        this->pNDArrayPool->convert(pArray, &intermediateIn, NDFloat32);
    }else{
        intermediateIn = pArray;
    }

    ssc_pimega_backend_restoration( (float *)pOutput->pData, (float *)intermediateIn->pData, &workspace );

    NDPluginDriver::endProcessCallbacks(pOutput, false, true);

    if (intermediateIn!=pArray){
        intermediateIn->release();
    }

}

asynStatus NDPluginSSCPimega::writeFloat64(asynUser *pasynUser, epicsFloat64 value)
{
    asynStatus ret;
    int function = pasynUser->reason;

    if(function<FIRST_NDPLUGIN_SSC_PIMEGA_PARAM)
        return NDPluginDriver::writeFloat64(pasynUser, value);

    int addr = 0;
    ret = getAddress(pasynUser, &addr);

    if(!ret) ret = setDoubleParam(addr, function, value);

    (void)callParamCallbacks(addr, addr);

    return ret;
}

asynStatus NDPluginSSCPimega::writeInt32(asynUser *pasynUser, epicsInt32 value)
{
    asynStatus ret;
    int function = pasynUser->reason;

    if(function<FIRST_NDPLUGIN_SSC_PIMEGA_PARAM)
        return NDPluginDriver::writeInt32(pasynUser, value);

    int addr = 0;
    ret = getAddress(pasynUser, &addr);

    if(function==loadMatrixes){
        //Should have the filePath configured before implementing this...
        getStringParam(matrixFilePath,256,matrixFileName);
        loadMatrix(value);
    }

    if(!ret) ret = setIntegerParam(addr, function, value);

    (void)callParamCallbacks(addr, addr);

    return ret;
}

extern "C" int NDSSCPimegaConfigure(const char *portName, int queueSize, int blockingCallbacks,
                                const char *NDArrayPort, int NDArrayAddr,
                                int maxBuffers, size_t maxMemory)
{
    return (new NDPluginSSCPimega(portName, queueSize, blockingCallbacks, NDArrayPort, NDArrayAddr,
                     maxBuffers, maxMemory, 0, 2000000))->start();
}

/* EPICS iocsh shell commands */
static const iocshArg initArg0 = { "portName",iocshArgString};
static const iocshArg initArg1 = { "frame queue size",iocshArgInt};
static const iocshArg initArg2 = { "blocking callbacks",iocshArgInt};
static const iocshArg initArg3 = { "NDArrayPort",iocshArgString};
static const iocshArg initArg4 = { "NDArrayAddr",iocshArgInt};
static const iocshArg initArg5 = { "maxBuffers",iocshArgInt};
static const iocshArg initArg6 = { "maxMemory",iocshArgInt};
static const iocshArg * const initArgs[] = {&initArg0,
                                            &initArg1,
                                            &initArg2,
                                            &initArg3,
                                            &initArg4,
                                            &initArg5,
                                            &initArg6};
static const iocshFuncDef initFuncDef = {"NDSSCPimegaConfigure",7,initArgs};
static void initCallFunc(const iocshArgBuf *args)
{
    NDSSCPimegaConfigure(args[0].sval, args[1].ival, args[2].ival,
                     args[3].sval, args[4].ival, args[5].ival,
                     args[6].ival);
}

static void NDSSCAPimegaRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" {
epicsExportRegistrar(NDSSCAPimegaRegister);
}
