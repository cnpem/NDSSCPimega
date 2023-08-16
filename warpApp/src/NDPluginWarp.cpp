/*
 * This software is Copyright by the Board of Trustees of Michigan
 * State University (c) Copyright 2016.
 *
 * Author: Michael Davidsaver <mdavidsaver@gmail.com>
 */
#include <algorithm>
#include <limits>

#include <string.h>

#include <epicsMath.h>

#include <iocsh.h>
#include <epicsExport.h>

#include <NDPluginWarp.h>

NDPluginWarp::NDPluginWarp(const char *portName, int queueSize, int blockingCallbacks,
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
                     asynInt32ArrayMask | asynFloat64ArrayMask | asynGenericPointerMask,
                     asynInt32ArrayMask | asynFloat64ArrayMask | asynGenericPointerMask,
                     0, // asynFlags
                     1, // autoConnect
                     priority,
                     stackSize,
                     1) // maxThreads
{
    lastinfo.nElements = (size_t)-1; // spoil

    createParam(NDWarpRunTimeString, asynParamFloat64, &NDWarpRunTime);
    createParam(NDWarpModeString, asynParamInt32, &NDWarpMode);
    createParam(NDWarpOutputString, asynParamInt32, &NDWarpOutput);

    setIntegerParam(NDWarpMode, 0); // nearest neighbor

    createParam(NDWarpAngleString, asynParamFloat64, &NDWarpAngle);
    createParam(NDWarpFactorXString, asynParamFloat64, &NDWarpFactorX);
    createParam(NDWarpFactorYString, asynParamFloat64, &NDWarpFactorY);
    createParam(NDWarpCenterXString, asynParamInt32, &NDWarpCenterX);
    createParam(NDWarpCenterYString, asynParamInt32, &NDWarpCenterY);

    setStringParam(NDPluginDriverPluginType, "NDPluginWarp");

    setDoubleParam(NDWarpFactorX, 0.0); // initialize w/ no-op
    setDoubleParam(NDWarpFactorY, 0.0);
    setDoubleParam(NDWarpAngle, 0.0);
    setIntegerParam(NDWarpCenterX, 0);
    setIntegerParam(NDWarpCenterY, 0);
}

NDPluginWarp::~NDPluginWarp() {
    assert(false); // never called (asyn ports can't be destory'd)
}

namespace {
template<typename T>
void warpit(NDArray *atemp,
            NDArray *output,
            const NDPluginWarp::Sample *S,
            size_t nElements,
            unsigned samp_per_pixel)
{
    const T * const  I = (const T*)atemp->pData;
    T       *        O = (T*)output->pData,
            * const OE = O+nElements;

    for(; O<OE; O++) {
        double val = 0.0;
        bool valid = true;
        for(unsigned j=0; j<samp_per_pixel; j++, S++) {
            valid &= S->valid;
            if(S->valid)
                val += S->weight * I[S->index];
        }
        if(!valid) val = 0.0;
        *O = (T)val;
    }

}
} // namespace

void
NDPluginWarp::processCallbacks(NDArray *pArray)
{
    // pArray is borrowed reference.  Caller will release()

    NDPluginDriver::beginProcessCallbacks(pArray);

    NDArrayInfo info;
    NDArray *pOutput;
    //(void)pArray->getInfo(&info);

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

    pOutput = this->pNDArrayPool->copy(pArray, NULL, true, true, true);
    
    (void)pOutput->getInfo(&info);
    printf("%lu\n",info.nElements);
    memset(pOutput->pData, 1, 4718592);

    NDPluginDriver::endProcessCallbacks(pOutput, true, true);

    pOutput->release();

}

asynStatus NDPluginWarp::writeFloat64(asynUser *pasynUser, epicsFloat64 value)
{
    asynStatus ret;
    int function = pasynUser->reason;

    if(function<FIRST_NDPLUGIN_WARP_PARAM)
        return NDPluginDriver::writeFloat64(pasynUser, value);

    int addr = 0;
    ret = getAddress(pasynUser, &addr);

    if(!ret) ret = setDoubleParam(addr, function, value);

    try {
        if(ret==asynSuccess) {
            lastinfo.nElements = (size_t)-1; // spoil
        }
    }catch(std::exception& e) {
        ret = asynError;
        epicsSnprintf(pasynUser->errorMessage, pasynUser->errorMessageSize,
                      "%s:: status=%d, addr=%d function=%d, value=%g : %s",
                      __PRETTY_FUNCTION__, ret, addr, function, value, e.what());

        asynPrint(pasynUser, ASYN_TRACE_ERROR,
                  "%s : Unhandled exception\n",
                  pasynUser->errorMessage);
    }

    (void)callParamCallbacks(addr, addr);

    return ret;
}

asynStatus NDPluginWarp::writeInt32(asynUser *pasynUser, epicsInt32 value)
{
    asynStatus ret;
    int function = pasynUser->reason;

    if(function<FIRST_NDPLUGIN_WARP_PARAM)
        return NDPluginDriver::writeInt32(pasynUser, value);

    int addr = 0;
    ret = getAddress(pasynUser, &addr);

    if(!ret) ret = setIntegerParam(addr, function, value);

    try {
        if(ret==asynSuccess) {
            lastinfo.nElements = (size_t)-1; // spoil
        }
    }catch(std::exception& e) {
        ret = asynError;
        epicsSnprintf(pasynUser->errorMessage, pasynUser->errorMessageSize,
                      "%s:: status=%d, addr=%d function=%d, value=%d : %s",
                      __PRETTY_FUNCTION__, ret, addr, function, value, e.what());

        asynPrint(pasynUser, ASYN_TRACE_ERROR,
                  "%s : Unhandled exception\n",
                  pasynUser->errorMessage);
    }

    (void)callParamCallbacks(addr, addr);

    return ret;
}

extern "C" int NDWarpConfigure(const char *portName, int queueSize, int blockingCallbacks,
                                const char *NDArrayPort, int NDArrayAddr,
                                int maxBuffers, size_t maxMemory)
{
    return (new NDPluginWarp(portName, queueSize, blockingCallbacks, NDArrayPort, NDArrayAddr,
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
static const iocshFuncDef initFuncDef = {"NDWarpConfigure",7,initArgs};
static void initCallFunc(const iocshArgBuf *args)
{
    NDWarpConfigure(args[0].sval, args[1].ival, args[2].ival,
                     args[3].sval, args[4].ival, args[5].ival,
                     args[6].ival);
}

static void NDWarpRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" {
epicsExportRegistrar(NDWarpRegister);
}
