
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <signal.h>
#include <ctype.h>

#include <caffe/caffe.hpp>
#include <caffe/solver.hpp>
#include <caffe/solver_factory.hpp>
#include <caffe/sgd_solvers.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/layers/sigmoid_cross_entropy_loss_layer.hpp>
#include <caffe/layers/euclidean_loss_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>

int requested_to_exit = 0;

using namespace std;
using namespace caffe;

double
get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);

	double time_in_sec = (tv.tv_sec) + ((double) tv.tv_usec * (double) 10e-7);
	return time_in_sec;
}


vector<caffe::Datum>
build_datum_vector(double x, double label)
{
	vector<caffe::Datum> datums;
	caffe::Datum d;

	d.set_channels(1);
	d.set_height(1);
	d.set_width(1);
	d.set_label(label);
	d.add_float_data(x);

	datums.push_back(d);
	return datums;
}


void
shutdown(int sign)
{
	if (sign == SIGINT)
	{
		printf("Exit requested\n");
		requested_to_exit = 1;
	}
}


int
main()
{
	signal(SIGINT, shutdown);

	boost::shared_ptr<caffe::Solver<float>> solver;
	boost::shared_ptr<caffe::Net<float>> net;

	caffe::SolverParameter solver_param;

	caffe::ReadProtoFromTextFileOrDie("solver.prototxt", &solver_param);
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

	net = solver->net();

	boost::shared_ptr<caffe::MemoryDataLayer<float>> input = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("input"));
	boost::shared_ptr<caffe::MemoryDataLayer<float>> target = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layer_by_name("target"));

	int epoch = 1;
	int sample = 1;

	double total_error_per_epoch = 0;

	double max_error = 0;
	int just_test = 0;

	double sampling_frequency = 1.0 / 20.0; // Hz
	double period_duration = 1.0; // second
	double curr_time = 0.0;

	while (epoch < 20000)
	{
		if (requested_to_exit)
			break;

		double x;

		x = (2 * M_PI * curr_time) / period_duration;

		double desired_y = sin(x);

		input->AddDatumVector(build_datum_vector(x, 0));
		target->AddDatumVector(build_datum_vector(desired_y, 0));

		float loss;

		if (!just_test)
		{
			solver->Step(1);
			loss = net->blob_by_name("loss")->cpu_data()[0];
		}
		else
		{
			net->Forward(&loss);
		}

		double estimated_y = net->blob_by_name("fc3")->cpu_data()[0];
		float diff = fabs(desired_y - estimated_y);

		if (just_test)
		{
			printf("TEST EPOCH %d SAMPLE %d LOSS: %.4f DES: %.4f EST: %.4f ERR: %.4f\n",
				epoch, sample, loss,
				desired_y, estimated_y, diff
			);
		}
		else
		{
			printf("TRAIN EPOCH %d SAMPLE %d LOSS: %.4f DES: %.4f EST: %.4f ERR: %.4f\n",
				epoch, sample, loss,
				desired_y, estimated_y, diff
			);
		}

		fflush(stdout);
		total_error_per_epoch += diff;

		if (diff > max_error)
			max_error = diff;

		sample++;
		curr_time += sampling_frequency;

		if (curr_time > period_duration)
		{
			printf("REPORT EPOCH %d total_err: %.4lf max_err: %.4lf\n",
					epoch, total_error_per_epoch, max_error);

			sample = 1;
			epoch++;
			total_error_per_epoch = 0.0;
			max_error = 0.0;
			curr_time = 0.0;

			just_test = !just_test;
		}
	}

	return 0;
}




