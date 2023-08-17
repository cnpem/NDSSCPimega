/*
 * This software is Copyright by the Board of Trustees of Michigan
 * State University (c) Copyright 2016.
 *
 * Author: Michael Davidsaver <mdavidsaver@gmail.com>
 */
#ifndef NDPLUGINWARP_H
#define NDPLUGINWARP_H

#include <memory>
#include <vector>

#include <epicsTypes.h>
#include <shareLib.h>

#include "NDPluginDriver.h"

/* Param definitions */

class epicsShareClass NDPluginWarp : public NDPluginDriver {
public:
    NDPluginWarp(const char *portName, int queueSize, int blockingCallbacks,
                 const char *NDArrayPort, int NDArrayAddr,
                 int maxBuffers, size_t maxMemory,
                 int priority, int stackSize);
    virtual ~NDPluginWarp();

    virtual void processCallbacks(NDArray *pArray);
    virtual asynStatus writeFloat64(asynUser *pasynUser, epicsFloat64 value);
    virtual asynStatus writeInt32(asynUser *pasynUser, epicsInt32 value);

    NDArrayInfo lastinfo;

    int NDWarpRunTime;

};

#endif // NDPLUGINWARP_H
