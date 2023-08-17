TOP=../..

include $(TOP)/configure/CONFIG
#----------------------------------------
#  ADD MACRO DEFINITIONS AFTER THIS LINE
#=============================

#=============================
# Build the IOC application

LIBRARY += NDPluginSSCPimega

DBD += NDPluginSSCPimega.dbd

# headers are not considered public

# Add all the support libraries needed by this IOC
NDPluginSSCPimega_LIBS += NDPlugin ADBase asyn xml2

NDPluginSSCPimega_SRCS += NDPluginSSCPimega.cpp

# Finally link to the EPICS Base libraries
NDPluginSSCPimega_LIBS += $(EPICS_BASE_IOC_LIBS)

#===========================

include $(TOP)/configure/RULES
#----------------------------------------
#  ADD RULES AFTER THIS LINE

