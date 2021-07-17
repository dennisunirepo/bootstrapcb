#include "Sink.h"

Define_Module(Sink);

void Sink::initialize() {
}

void Sink::handleMessage(cMessage *msg) {
    EV << "Received " << msg->getName() << endl;
	delete msg;
}
