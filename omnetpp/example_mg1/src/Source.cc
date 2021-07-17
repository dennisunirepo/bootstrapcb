#include "Source.h"

Define_Module(Source);

Source::Source() {
	sendMessageEvent = nullptr;
}

Source::~Source() {
	cancelAndDelete(sendMessageEvent);
}

void Source::initialize() {
	sendMessageEvent = new cMessage("sendMessageEvent");
    // schedule first arrival
	scheduleAt(simTime(), sendMessageEvent);
}

void Source::handleMessage(cMessage *msg) {
	ASSERT(msg==sendMessageEvent);
	// generate new arrival
    Job *arrival = new Job("arrival");
	send(arrival, "out");
	// determine next arrival time
	simtime_t nextInterarrivalTime = par("interArrivalTime");
	// schedule next arrival
	scheduleAt(simTime()+nextInterarrivalTime, sendMessageEvent);
}
