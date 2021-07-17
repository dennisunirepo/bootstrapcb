package example_mg1;

simple Server {
	parameters:
		@display("i=block/queue");
		@signal[iarrivalTime](type=simtime_t);
		@signal[serviceTime](type=simtime_t);
        @signal[queueLength](type=double);
        @signal[nCustomers](type=double);
        @signal[queueingTime](type=simtime_t);
        @signal[waitingTime](type=simtime_t);
        @statistic[iarrivalTime](title="inter arrival time";record=vector?,mean?,timeavg?,histogram?);
        @statistic[serviceTime](title="service time";record=vector?,mean?,timeavg?,histogram?);
        @statistic[queueLength](title="queue length";record=vector?,mean?,timeavg?,histogram?);
        @statistic[nCustomers](title="customers in system";record=vector?,mean?,timeavg?,histogram?);
        @statistic[queueingTime](title="time spent in queue";record=vector?,mean?,timeavg?,histogram?);
        @statistic[waitingTime](title="queueing time at dequeue";record=vector?,mean?,timeavg?,histogram?);
		volatile double serviceTime @unit(s);
		int buffer = default(-1);
    gates:
        input in;
        output out;
}