import os 
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys, os
from contextlib import contextmanager
from scipy.stats import chi2, rv_discrete, rv_continuous

class genericAnalysis():
    def __init__(self, sim_runs_dataset, bootstrapcb_dataset, simstudy_dataset,
                 figure_path, model_class, theta_0, df=None):
        # parametric model
        self.theta_0 = theta_0
        self.df = len(self.theta_0) if theta_0 is not None and df is None else df
        self.model_class = model_class
        # files
        self.figure_path = figure_path
        self.sim_runs_dataset = sim_runs_dataset
        self.bootstrapcb_dataset = bootstrapcb_dataset
        self.simstudy_dataset = simstudy_dataset
        # plots
        self.figure_couter = 0
        self.figures = []

    def ecdf(self, x, X):
        return np.array([len(X[(X < x_i)])/len(X) for x_i in x])
        
    def get_new_fig(self, **kwds):
        self.figure_couter += 1
        fig = plt.figure(self.figure_couter, **kwds)
        self.figures.append(fig)
        return fig

    def get_information_for_dataset(self):
        nruns = len(self.fitresults_per_run)
        tmp_run = self.fitresults_per_run[0]
        nsamples = len(tmp_run['endog'])
        nreps = len(tmp_run['Theta_bs'])
        print('###############################################################')
        print('## Stored information #########################################')
        print('###############################################################')
        items = ''
        for item in tmp_run:
            items += '{},'.format(item)
        print(items[:-1])
        print('nruns: {} \nsamples: {} \nnreps: {}'.format(nruns, nsamples, nreps))

    def save_figures(self):        
        for i, fig in enumerate(self.figures):
            fig.savefig(self.figure_path + 'fig{}.png'.format(i))
    
    def save_dict_to_file(self, path, dict):
        data = json.dumps(dict)
        file = open(path, "w")
        file.write(data)
        file.close() 

    def get_dict_from_file(self, path):
        file = open(path, "r")
        data = file.read()
        file.close() 
        dict = json.loads(data)
        return dict

    def data_plots(self, nx=50, xmin=None, xmax=None):
        if self.results.model.timestamps is not None:
            fig = self.get_new_fig()
            step_timedata = fig.add_subplot(111)
            step_timedata.plot(self.results.model.timestamps, self.results.model.endog, drawstyle='steps-post')
            step_timedata.legend()
            step_timedata.set_title('time series')
        # get distributions
        if sum(self.results.model.exog) != 0:
            fig = self.get_new_fig()
            scatter = fig.add_subplot(111)
            #
            if (xmin is None) or (xmax is None):
                xmin = np.min(self.results.model.exog)
                xmax = np.max(self.results.model.exog)
            x_0 = self.results.model.exog
            data_0 = self.results.model.endog
            xs = np.linspace(xmin, xmax, nx)
            fitted_model = self.results.model.display_func(xs, *self.theta_0)
            # do plots 
            scatter.plot(x_0, data_0, 'o', label='data') 
            scatter.plot(xs, fitted_model, label='fitted model')
            # configure plots
            scatter.legend()
            scatter.set_title('data and fitted model')
        else:        
            fig = self.get_new_fig()
            hist_cdf = fig.add_subplot(111)
            fig2 = self.get_new_fig()
            hist_pdf = fig2.add_subplot(111)
            #
            if (xmin is None) or (xmax is None):
                xmin = np.min(self.results.model.endog)
                xmax = np.max(self.results.model.endog)
            if isinstance(self.results.model.dist, rv_continuous):
                xs = np.linspace(xmin, xmax, nx)
                ecdf = self.ecdf(xs, self.results.model.endog) 
                F_0 = self.results.model.cdf(xs, *self.theta_0)
                f_0 = self.results.model.pdf(xs, *self.theta_0)
            elif isinstance(self.results.model.dist, rv_discrete):
                xs = np.arange(np.min(self.results.model.endog), np.max(self.results.model.endog)+1)
                ecdf = self.ecdf(xs, self.results.model.endog) 
                F_0 = self.results.model.cdf(xs, *self.theta_0)
                f_0 = self.results.model.pmf(xs, *self.theta_0)
            # do plots 
            hist_cdf.plot(xs, ecdf, drawstyle='steps-post', label='ECDF')
            hist_cdf.plot(xs, F_0, label='true CDF') 
            hist_pdf.hist(self.results.model.endog, density=True, cumulative=False, label='histogram')
            hist_pdf.plot(xs, f_0, label='true PDF')
            # configure plots
            hist_cdf.legend()
            hist_cdf.set_title('cumulative distribution function')
            hist_pdf.legend()
            hist_pdf.set_title('probability density function')
        # print out data
        print("Observed data:")
        for _,d in enumerate(self.results.model.endog):
            print(d)
        print("theta0: {}".format(self.theta_0))
        print("Fitted model:")
        print("theta: {}".format(self.results.params))
        print("cov: {}".format(self.results.normalized_cov_params))
        return fig

    def confregion_plots(self, xmin=None, xmax=None, ymin=None, ymax=None,
                         px=0, py=1):   
        # build plots
        fig = self.get_new_fig()
        likelihood  = fig.add_subplot(111, projection='3d')
        fig = self.get_new_fig()
        scatter_bs = fig.add_subplot(111)
        fig = self.get_new_fig()
        plot3d_w = fig.add_subplot(111, projection='3d')
        fig = self.get_new_fig()
        contour_w = fig.add_subplot(111)
        fig = self.get_new_fig()
        contour_w_bs = fig.add_subplot(111)
        fig = self.get_new_fig()
        contour_studw_bs = fig.add_subplot(111)
        fig = self.get_new_fig()
        plot3d_lr = fig.add_subplot(111, projection='3d')
        fig = self.get_new_fig()
        contour_lr = fig.add_subplot(111)
        fig = self.get_new_fig()
        contour_lr_bs = fig.add_subplot(111)
        fig = self.get_new_fig()
        hist_t_dist  = fig.add_subplot(111)
        #
        p = np.copy(self.results.params)
        cmap = plt.get_cmap('gist_earth')
        #
        if (xmin is None) or (xmax is None):
            xmin = self.results.params[px] - 5*np.sqrt(self.results.normalized_cov_params[px,px])
            xmax = self.results.params[px] + 5*np.sqrt(self.results.normalized_cov_params[px,px])
        if (ymin is None) or (ymax is None):
            ymin = self.results.params[py] - 5*np.sqrt(self.results.normalized_cov_params[py,py])
            ymax = self.results.params[py] + 5*np.sqrt(self.results.normalized_cov_params[py,py])
        xs               = np.linspace(xmin, xmax, 50)
        ys               = np.linspace(ymin, ymax, 50)
        X, Y = np.meshgrid(xs, ys)
        # 3d likelihood
        Z = np.zeros_like(X)
        Z_lr = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(Y)):
                p = np.copy(self.results.params)
                p[px] = X[i][j]
                p[py] = Y[i][j]
                Z[i][j] = self.results.model.loglike(p)
                Z_lr[i][j] = self.results.likelihood_ratio_statistic(p)
        #
        p[px] = np.min(X)
        p[py] = np.min(Y)
        z1 = self.results.model.loglike(p)
        p[px] = np.min(X)
        p[py] = np.max(Y)
        z2 = self.results.model.loglike(p)
        p[px] = np.max(X)
        p[py] = np.min(Y)
        z3 = self.results.model.loglike(p)
        p[px] = np.max(X)
        p[py] = np.max(Y)
        z4 = self.results.model.loglike(p)
        z_min = min(z1, z2, z3, z4)
        # 
        z_mle = self.results.model.loglike(self.results.params)
        z_0 = self.results.model.loglike(self.theta_0)
        # 
        likelihood.plot_surface(X, Y, Z, cmap=cmap, alpha=0.7)
        likelihood.plot([self.results.params[px], np.min(X)],
                        [self.results.params[py], self.results.params[py]],
                        [z_mle, z_mle], linestyle='--', color='grey')
        likelihood.plot([self.results.params[px], self.results.params[px]],
                        [self.results.params[py], np.max(Y)],
                        [z_mle, z_mle], linestyle='--', color='grey')
        likelihood.plot([self.results.params[px], self.results.params[px]],
                        [self.results.params[py], self.results.params[py]],
                        [z_min, z_mle], linestyle='--', color='grey')
        likelihood.plot([self.results.params[px]],
                        [self.results.params[py]],
                        [z_mle], 'o', label='MLE: {}'.format([np.round(self.results.params[px], 2), np.round(self.results.params[py], 2)]))
        likelihood.plot([self.theta_0[px], np.min(X)],
                        [self.theta_0[py], self.theta_0[py]],
                        [z_0, z_0], linestyle='--', color='grey')
        likelihood.plot([self.theta_0[px], self.theta_0[px]],
                        [self.theta_0[py], np.max(Y)],
                        [z_0, z_0], linestyle='--', color='grey')
        likelihood.plot([self.theta_0[px], self.theta_0[px]],
                        [self.theta_0[py], self.theta_0[py]],
                        [z_min, z_0], linestyle='--', color='grey')
        likelihood.plot([self.theta_0[px]],
                        [self.theta_0[py]],
                        [z_0], 'o', label='theta0: {}'.format([self.theta_0[px], self.theta_0[py]]))
        # 3d likelihood ratio
        p[px] = np.min(X)
        p[py] = np.min(Y)
        z1 = self.results.likelihood_ratio_statistic(p)[0]
        p[px] = np.min(X)
        p[py] = np.max(Y)
        z2 = self.results.likelihood_ratio_statistic(p)[0]
        p[px] = np.max(X)
        p[py] = np.min(Y)
        z3 = self.results.likelihood_ratio_statistic(p)[0]
        p[px] = np.max(X)
        p[py] = np.max(Y)
        z4 = self.results.likelihood_ratio_statistic(p)[0]
        z_lr_min = min(z1, z2, z3, z4)
        #
        z_lr_mle = self.results.likelihood_ratio_statistic(self.results.params)[0]
        z_lr_0 = self.results.likelihood_ratio_statistic(self.theta_0)[0]
        #
        plot3d_lr.plot_surface(X, Y, Z_lr, cmap=cmap, alpha=0.7)
        plot3d_lr.plot([self.results.params[px], np.min(X)],
                       [self.results.params[py], self.results.params[py]],
                       [z_lr_mle, z_lr_mle], linestyle='--', color='grey')
        plot3d_lr.plot([self.results.params[px], self.results.params[px]],
                       [self.results.params[py], np.max(Y)],
                       [z_lr_mle, z_lr_mle], linestyle='--', color='grey')
        plot3d_lr.plot([self.results.params[px], self.results.params[px]],
                       [self.results.params[py], self.results.params[py]],
                       [z_lr_min, z_lr_mle], linestyle='--', color='grey')
        plot3d_lr.plot([self.results.params[px]],
                       [self.results.params[py]],
                       [z_lr_mle], 'o', label='MLE: {}'.format([np.round(self.results.params[px], 2), np.round(self.results.params[py], 2)]))
        plot3d_lr.plot([self.theta_0[px], np.min(X)],
                       [self.theta_0[py], self.theta_0[py]],
                       [z_lr_0, z_lr_0], linestyle='--', color='grey')
        plot3d_lr.plot([self.theta_0[px], self.theta_0[px]],
                       [self.theta_0[py], np.max(Y)],
                       [z_lr_0, z_lr_0], linestyle='--', color='grey')
        plot3d_lr.plot([self.theta_0[px], self.theta_0[px]],
                       [self.theta_0[py], self.theta_0[py]],
                       [z_lr_min, z_lr_0], linestyle='--', color='grey')
        plot3d_lr.plot([self.theta_0[px]],
                       [self.theta_0[py]],
                       [z_lr_0], 'o', label='theta0: {}'.format([self.theta_0[px], self.theta_0[py]]))
        # 3d wald statistik
        Z_w = np.zeros_like(X)
        for i in range(len(X)):
            for j in range(len(Y)):
                p[px] = X[i][j]
                p[py] = Y[i][j]
                Z_w[i][j]  = self.results.wald_statistic(p)
        #
        p[px] = np.min(X)
        p[py] = np.min(Y)
        z1 = self.results.wald_statistic(p)[0]
        p[px] = np.min(X)
        p[py] = np.max(Y)
        z2 = self.results.wald_statistic(p)[0]
        p[px] = np.max(X)
        p[py] = np.min(Y)
        z3 = self.results.wald_statistic(p)[0]
        p[px] = np.max(X)
        p[py] = np.max(Y)
        z4 = self.results.wald_statistic(p)[0]
        z_w_min = min(z1, z2, z3, z4)
        # 
        z_w_mle = self.results.wald_statistic(self.results.params)[0]
        z_w_0 = self.results.wald_statistic(self.theta_0)[0]
        #
        plot3d_w.plot_surface(X, Y, Z_w, cmap=cmap, alpha=0.7)
        plot3d_w.plot([self.results.params[px], np.min(X)],
                       [self.results.params[py], self.results.params[py]],
                       [z_w_mle, z_w_mle], linestyle='--', color='grey')
        plot3d_w.plot([self.results.params[px], self.results.params[px]],
                       [self.results.params[py], np.max(Y)],
                       [z_w_mle, z_w_mle], linestyle='--', color='grey')
        plot3d_w.plot([self.results.params[px], self.results.params[px]],
                       [self.results.params[py], self.results.params[py]],
                       [z_w_min, z_w_mle], linestyle='--', color='grey')
        plot3d_w.plot([self.results.params[px]],
                       [self.results.params[py]],
                       [z_w_mle], 'o', label='MLE: {}'.format([np.round(self.results.params[px], 2), np.round(self.results.params[py], 2)]))
        plot3d_w.plot([self.theta_0[px], np.min(X)],
                       [self.theta_0[py], self.theta_0[py]],
                       [z_w_0, z_w_0], linestyle='--', color='grey')
        plot3d_w.plot([self.theta_0[px], self.theta_0[px]],
                       [self.theta_0[py], np.max(Y)],
                       [z_w_0, z_w_0], linestyle='--', color='grey')
        plot3d_w.plot([self.theta_0[px], self.theta_0[px]],
                       [self.theta_0[py], self.theta_0[py]],
                       [z_w_min, z_w_0], linestyle='--', color='grey')
        plot3d_w.plot([self.theta_0[px]],
                       [self.theta_0[py]],
                       [z_w_0], 'o', label='theta0: {}'.format([self.theta_0[px], self.theta_0[py]]))
        # mle 
        mle              = np.copy(self.results.params)
        chi2_levels      = chi2.ppf([0.5,0.95,0.99], df=self.df)
        Theta            = self.results._params_uniform_on_confregion(chi2_levels[1], nsamples=100)
        # tests 
        T_w_bs              = self.results.wald_statistic(self.results.Params_bs)
        T_studw_bs          = self.results.wald_statistic(self.results.Params_bs, self.results.Normalized_cov_params_bs)
        T_lr_bs             = self.results.likelihood_ratio_statistic(self.results.Params_bs)
        # levels for percentiles
        T_w_bs_levels       = self.results.percentile(T_w_bs, alpha=[0.5,0.05,0.01]) 
        T_studw_bs_levels   = self.results.percentile(T_studw_bs, alpha=[0.5,0.05,0.01]) 
        T_lr_bs_levels      = self.results.percentile(T_lr_bs, alpha=[0.5,0.05,0.01]) 
        # do other plots
        scatter_bs.plot(self.results.Params_bs[:, 0], self.results.Params_bs[:, 1], 'o', label='bs samples')
        if len(self.results.params) < 3: scatter_bs.plot(Theta[:, 0], Theta[:, 1], 'o', label='samples uniform on sphere')
        scatter_bs.plot(mle[0], mle[1], 'o', label='mle')
        scatter_bs.plot(self.theta_0[0], self.theta_0[1], 'o', label='theta0')
        CS_bs = scatter_bs.contour(X, Y, Z_w, levels=chi2_levels, colors='grey', linestyles='--', labels='test')
        #
        contour_w.plot(self.results.Params_bs[:, 0], self.results.Params_bs[:, 1], 'o', label='bs samples', alpha=0.5)
        # if len(self.results.params) < 3: contour_w.plot(Theta[:, 0], Theta[:, 1], 'o', label='samples uniform on sphere')
        contour_w.plot(mle[0], mle[1], 'o', label='mle')
        contour_w.plot(self.theta_0[0], self.theta_0[1], 'o', label='theta0')
        CS_w = contour_w.contour(X, Y, Z_w,  levels=chi2_levels, colors='grey', linestyles='--') 
        #
        contour_w_bs.plot(self.results.Params_bs[:, 0], self.results.Params_bs[:, 1], 'o', label='bs samples', alpha=0.5)
        # if len(self.results.params) < 3: contour_w_bs.plot(Theta[:, 0], Theta[:, 1], 'o', label='samples uniform on sphere')
        contour_w_bs.plot(mle[0], mle[1], 'o', label='mle')
        contour_w_bs.plot(self.theta_0[0], self.theta_0[1], 'o', label='theta0')
        CS_w_bs = contour_w_bs.contour(X, Y, Z_w,  levels=T_w_bs_levels, colors='grey', linestyles='--') # green
        #
        contour_studw_bs.plot(self.results.Params_bs[:, 0], self.results.Params_bs[:, 1], 'o', label='bs samples', alpha=0.5)
        # if len(self.results.params) < 3: contour_studw_bs.plot(Theta[:, 0], Theta[:, 1], 'o', label='samples uniform on sphere')
        contour_studw_bs.plot(mle[0], mle[1], 'o', label='mle')
        contour_studw_bs.plot(self.theta_0[0], self.theta_0[1], 'o', label='theta0')
        CS_studw_bs = contour_studw_bs.contour(X, Y, Z_w,  levels=T_studw_bs_levels, colors='grey', linestyles='--') # cyan
        #
        contour_lr.plot(self.results.Params_bs[:, 0], self.results.Params_bs[:, 1], 'o', label='bs samples', alpha=0.5)
        # if len(self.results.params) < 3: contour_lr.plot(Theta[:, 0], Theta[:, 1], 'o', label='samples uniform on sphere')
        contour_lr.plot(mle[0], mle[1], 'o', label='mle')
        contour_lr.plot(self.theta_0[0], self.theta_0[1], 'o', label='theta0')
        CS_lr = contour_lr.contour(X, Y, Z_lr, levels=chi2_levels, colors='grey', linestyles='--') #magenta
        #
        contour_lr_bs.plot(self.results.Params_bs[:, 0], self.results.Params_bs[:, 1], 'o', label='bs samples', alpha=0.5)
        # if len(self.results.params) < 3: contour_lr_bs.plot(Theta[:, 0], Theta[:, 1], 'o', label='samples uniform on sphere')
        contour_lr_bs.plot(mle[0], mle[1], 'o', label='mle')
        contour_lr_bs.plot(self.theta_0[0], self.theta_0[1], 'o', label='theta0')
        CS_lr_bs = contour_lr_bs.contour(X, Y, Z_lr, levels=T_lr_bs_levels, colors='grey', linestyles='--') 
        #
        xs = np.linspace(0, chi2.ppf(0.99, df=self.df))
        hist_t_dist.plot(xs, chi2.cdf(xs, df=self.df), label='Chi2(2)')
        hist_t_dist.plot(xs, self.ecdf(xs, T_w_bs),      color='green',   drawstyle='steps-post', label='bs wald-test')
        hist_t_dist.plot(xs, self.ecdf(xs, T_studw_bs), color='cyan',    drawstyle='steps-post', label='bs studentized wald-test')
        hist_t_dist.plot(xs, self.ecdf(xs, T_lr_bs),   color='magenta', drawstyle='steps-post', label='bs likelihood ratio')
        # configure plots
        scatter_bs.set_xlim(xmin, xmax)
        scatter_bs.set_ylim(ymin, ymax)
        contour_w.set_xlim(xmin, xmax)
        contour_w.set_ylim(ymin, ymax)
        contour_w_bs.set_xlim(xmin, xmax)
        contour_w_bs.set_ylim(ymin, ymax)
        contour_studw_bs.set_xlim(xmin, xmax)
        contour_studw_bs.set_ylim(ymin, ymax)
        contour_lr.set_xlim(xmin, xmax)
        contour_lr.set_ylim(ymin, ymax)
        contour_lr_bs.set_xlim(xmin, xmax)
        contour_lr_bs.set_ylim(ymin, ymax)
        #    
        scatter_bs.clabel(CS_bs, CS_bs.levels, inline=True, fmt=self.fmt, fontsize=10)
        contour_w.clabel(CS_w, CS_w.levels, inline=True, fmt=self.fmt, fontsize=10)
        contour_w_bs.clabel(CS_w_bs, CS_w_bs.levels, inline=True, fmt=self.fmt, fontsize=10)
        contour_studw_bs.clabel(CS_studw_bs, CS_studw_bs.levels, inline=True, fmt=self.fmt, fontsize=10)
        contour_lr.clabel(CS_lr, CS_lr.levels, inline=True, fmt=self.fmt, fontsize=10)
        contour_lr_bs.clabel(CS_lr_bs, CS_lr_bs.levels, inline=True, fmt=self.fmt, fontsize=10)
        #
        likelihood.legend()
        likelihood.set_title('likelihood')
        scatter_bs.legend()
        scatter_bs.set_title('bootstrap samples')
        plot3d_w.legend()
        plot3d_w.set_title('wald statistic')
        contour_w.legend()
        contour_w.set_title('wald statistic quantiles')
        contour_w_bs.legend()
        contour_w_bs.set_title('bootstrapped wald statistic quantiles')
        contour_studw_bs.legend()
        contour_studw_bs.set_title('bootstrapped studentized wald quantiles')
        plot3d_lr.legend()
        plot3d_lr.set_title('likelihoodratio statistic')
        contour_lr.legend()
        contour_lr.set_title('likelihoodratio quantiles')
        contour_lr_bs.legend()
        contour_lr_bs.set_title('bootstrapped likelihoodratio quantiles')
        hist_t_dist.legend()
        hist_t_dist.set_title('chi2 distribution')
        return fig

    def confband_plots(self, nx=50, xmin=None, xmax=None, ymin=0, ymax=1):
        # true cdf or data
        if sum(self.results.model.exog) != 0:
            if (xmin is None) or (xmax is None):
                xmin = np.min(self.results.model.exog)
                xmax = np.max(self.results.model.exog)
            xs = np.linspace(xmin, xmax, nx)
            x_0 = self.results.model.exog
            data_0 = self.results.model.endog
            data_0_formatstr = 'o'
            data_0_drawstyle = None
        else:        
            if (xmin is None) or (xmax is None):
                xmin = np.min(self.results.model.endog)
                xmax = np.max(self.results.model.endog)
            if isinstance(self.results.model.dist, rv_continuous):
                xs = np.linspace(xmin, xmax, nx)
            elif isinstance(self.results.model.dist, rv_discrete):
                xs = np.arange(xmin, xmax+1)
            x_0 = np.linspace(xmin, xmax, len(self.results.model.endog)+2)
            x_0[1:-1] = np.sort(self.results.model.endog)
            data_0 = self.results.model.ecdf(x_0)
            data_0_formatstr = ''
            data_0_drawstyle = 'steps-post'
        
        # mle
        mle                = self.results.mle(xs)
        
        # bootstrap statistics
        T_bs_w             = self.results.wald_statistic(self.results.Params_bs)
        # T_bs_stud          = self.results.wald_statistic(self.results.Params_bs, self.results.Normalized_cov_params_bs)
        # T_bs_lr            = self.results.likelihood_ratio_statistic(self.results.Params_bs)
        
        # confidence bands
        confint_delta           = self.results.confint(xs, alpha=0.1, method='delta')
        # confint_bs              = self.results.confint(xs, alpha=0.1, method='bs')
        # confint_bst             = self.results.confint(xs, alpha=0.1, method='bst')
        confband_bonf           = self.results.confband(xs, alpha=0.1, method='bonf')
        # confband_bonf_bs        = self.results.confband(xs, alpha=0.1, method='bonf_bs')
        confband_ks           = self.results.confband(xs, alpha=0.1, method='ks')
        confband_ks_bs        = self.results.confband(xs, alpha=0.1, method='ks_bs')
        confband_delta          = self.results.confband(xs, alpha=0.1, test='asymp', df=self.df, method='delta')
        # confband_delta_bs       = self.results.confband(xs, alpha=0.1, T=T_bs_w, method='delta')
        confband_mc             = self.results.confband(xs, alpha=0.1, test='asymp', df=self.df, method='mc')
        confband_mc_bs          = self.results.confband(xs, alpha=0.1, T=T_bs_w, method='mc')
        if True:
            confband_nm             = self.results.confband(xs, alpha=0.1, test='asymp', df=self.df, method='nm')
            confband_nm_bs          = self.results.confband(xs, alpha=0.1, test='w', method='nm')
            confband_nm_bst         = self.results.confband(xs, alpha=0.1, test='studw', method='nm')
        if True:
            confband_lagr_bs        = self.results.confband(xs, alpha=0.1, test='lr', method='lagrange')
        # confband_lagr_lr_bs     = self.results.confband(xs, alpha=0.1, test='lr', method='lagrange')
        # confband_per_bs_data    = self.results.confband(xs, alpha=0.1, test='w', method='data')
       
        # do plots
        if True:
            fig = self.get_new_fig()
            pwiseband = fig.add_subplot(111)
            pwiseband.plot(x_0, data_0, data_0_formatstr, drawstyle=data_0_drawstyle, color='blue', label='data')
            pwiseband.plot(xs,      mle,            color='orange',  label='mle')
            pwiseband.plot(xs,      confint_delta[:, 0],  color='red', linestyle='--',   label='ci delta method')
            pwiseband.plot(xs,      confint_delta[:, 1],  color='red', linestyle='--')
            # pwiseband.plot(xs,      confint_bs[:, 0],  color='red', label='ci bootstrap')
            # pwiseband.plot(xs,      confint_bs[:, 1],  color='red')
            # pwiseband.plot(xs,      confint_bst[:, 0],  color='green', label='ci bootstrap-t')
            # pwiseband.plot(xs,      confint_bst[:, 1],  color='green')
            if (ymin is not None) and (ymax is not None):
                pwiseband.set_ylim(ymin, ymax)
            if (xmin is not None) and (xmax is not None):
                pwiseband.set_xlim(xmin, xmax)
            pwiseband.legend()
            pwiseband.set_title('confidence intervals')
        #
        if True:
            fig = self.get_new_fig()
            band_asymp = fig.add_subplot(111)
            band_asymp.plot(x_0, data_0, data_0_formatstr, drawstyle=data_0_drawstyle, color='blue', label='data')
            band_asymp.plot(xs,      mle,            color='orange',  label='mle')
            band_asymp.plot(xs,      confint_delta[:, 0],  color='red', linestyle='--',   label='ci delta method')
            band_asymp.plot(xs,      confint_delta[:, 1],  color='red', linestyle='--')
            band_asymp.plot(xs,      confband_delta[:, 0],  color='green', linestyle='--',  label='cb delta method')
            band_asymp.plot(xs,      confband_delta[:, 1],  color='green', linestyle='--')
            band_asymp.plot(xs,      confband_mc[:, 0],  color='green',  label='cb asymp mc')
            band_asymp.plot(xs,      confband_mc[:, 1],  color='green')
            band_asymp.plot(xs,      confband_nm[:, 0],  color='magenta', label='cb asymp')
            band_asymp.plot(xs,      confband_nm[:, 1],  color='magenta')
            if (ymin is not None) and (ymax is not None):
                band_asymp.set_ylim(ymin, ymax)
            if (xmin is not None) and (xmax is not None):
                band_asymp.set_xlim(xmin, xmax)
            band_asymp.legend()
            band_asymp.set_title('asymptotic confidence bands')
        #
        if True:
            fig = self.get_new_fig()
            band_vgl = fig.add_subplot(111)
            band_vgl.plot(x_0, data_0, data_0_formatstr, drawstyle=data_0_drawstyle, color='blue',    label='data')
            band_vgl.plot(xs,      mle,             color='orange',  label='mle')
            band_vgl.plot(xs,      confint_delta[:, 0],  color='red', linestyle='--',   label='ci delta method')
            band_vgl.plot(xs,      confint_delta[:, 1],  color='red', linestyle='--')
            band_vgl.plot(xs,      confband_delta[:, 0],  color='green', linestyle='--',  label='cb delta method')
            band_vgl.plot(xs,      confband_delta[:, 1],  color='green', linestyle='--')
            band_vgl.plot(xs,      confband_mc_bs[:, 0],  color='green', label='cb bootstrap mc')
            band_vgl.plot(xs,      confband_mc_bs[:, 1],  color='green')
            band_vgl.plot(xs,      confband_nm_bs[:, 0],  color='magenta',   label='cb bootstrap')
            band_vgl.plot(xs,      confband_nm_bs[:, 1],  color='magenta')
            band_vgl.plot(xs,      confband_nm_bst[:, 0], color='cyan',    label='cb bootstrap-t')
            band_vgl.plot(xs,      confband_nm_bst[:, 1], color='cyan')
            # band_vgl.plot(xs,      confband_lagr_lr_bs[:, 0], color='pink',    label='cb likelihoodratio')
            # band_vgl.plot(xs,      confband_lagr_lr_bs[:, 1], color='pink')
            if (ymin is not None) and (ymax is not None):
                band_vgl.set_ylim(ymin, ymax)
            if (xmin is not None) and (xmax is not None):
                band_vgl.set_xlim(xmin, xmax)
            band_vgl.legend()
            band_vgl.set_title('bootstrap confidence bands')
        #
        if True:
            fig = self.get_new_fig()
            band_bonfvgl = fig.add_subplot(111)
            band_bonfvgl.plot(x_0, data_0, data_0_formatstr, drawstyle=data_0_drawstyle, color='blue',    label='data')
            band_bonfvgl.plot(xs,      mle,            color='orange',  label='mle')
            band_bonfvgl.plot(xs,      confband_bonf[:, 0],  color='red', linestyle='--', label='cb bonferroni')
            band_bonfvgl.plot(xs,      confband_bonf[:, 1],  color='red', linestyle='--')
            band_bonfvgl.plot(xs,      confband_ks[:, 0],  color='green', linestyle='--', label='cb ks')
            band_bonfvgl.plot(xs,      confband_ks[:, 1],  color='green', linestyle='--')
            band_bonfvgl.plot(xs,      confband_mc_bs[:, 0],     color='green', label='cb bootstrap mc')
            band_bonfvgl.plot(xs,      confband_mc_bs[:, 1],     color='green')
            band_bonfvgl.plot(xs,      confband_nm_bs[:, 0],     color='magenta',   label='cb bootstrap')
            band_bonfvgl.plot(xs,      confband_nm_bs[:, 1],     color='magenta')
            band_bonfvgl.plot(xs,      confband_nm_bst[:, 0], color='cyan',    label='cb bootstrap-t')
            band_bonfvgl.plot(xs,      confband_nm_bst[:, 1], color='cyan')
            # band_bonfvgl.plot(xs,      confband_lagr_lr_bs[:, 0], color='pink',    label='cb likelihoodratio')
            # band_bonfvgl.plot(xs,      confband_lagr_lr_bs[:, 1], color='pink')
            if (ymin is not None) and (ymax is not None):
                band_bonfvgl.set_ylim(ymin, ymax)
            if (xmin is not None) and (xmax is not None):
                band_bonfvgl.set_xlim(xmin, xmax)
            band_bonfvgl.legend()
            band_bonfvgl.set_title('bootstrap vs classic confidence bands')
        #
        if True:
            fig = self.get_new_fig()
            band_bsbonfks = fig.add_subplot(111)
            band_bsbonfks.plot(x_0, data_0, data_0_formatstr, drawstyle=data_0_drawstyle, color='blue',    label='data')
            band_bsbonfks.plot(xs,      mle,            color='orange',  label='mle')
            band_bsbonfks.plot(xs,      confint_delta[:, 0],  color='red', linestyle='--',   label='ci delta method')
            band_bsbonfks.plot(xs,      confint_delta[:, 1],  color='red', linestyle='--')
            band_bsbonfks.plot(xs,      confband_delta[:, 0],  color='green', linestyle='--',  label='cb delta method')
            band_bsbonfks.plot(xs,      confband_delta[:, 1],  color='green', linestyle='--')
            # band_bsbonfks.plot(xs,      confband_bonf_bs[:, 0],  color='red', label='cb bonferroni bootstrapped')
            # band_bsbonfks.plot(xs,      confband_bonf_bs[:, 1],  color='red')
            band_bsbonfks.plot(xs,      confband_ks_bs[:, 0],  color='green', label='cb ks bootstrapped')
            band_bsbonfks.plot(xs,      confband_ks_bs[:, 1],  color='green')
            band_bsbonfks.plot(xs,      confband_lagr_bs[:, 0],  color='magenta',  label='cb bootstrap lr')
            band_bsbonfks.plot(xs,      confband_lagr_bs[:, 1],  color='magenta')
            if (ymin is not None) and (ymax is not None):
                band_bsbonfks.set_ylim(ymin, ymax)
            if (xmin is not None) and (xmax is not None):
                band_bsbonfks.set_xlim(xmin, xmax)
            band_bsbonfks.legend()
            band_bsbonfks.set_title('other bootstrap confidence bands')

    @contextmanager
    def suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout

    def parse_if_number(self, s):
        try: return float(s)
        except: return True if s=="true" else False if s=="false" else s if s else None

    def parse_ndarray(self, s):
        return np.fromstring(s, sep=' ') if s else None

    def load_data_from_sim_dataset(self, statistic, strategy, nbatches):
        # get dataset  
        df = pd.read_csv(self.sim_runs_dataset, 
            converters = {
                'attrvalue': self.parse_if_number,
                'binedges': self.parse_ndarray,
                'binvalues': self.parse_ndarray,
                'vectime': self.parse_ndarray,
                'vecvalue': self.parse_ndarray
            })
        print(df.head()) # print an excerpt of the result
        # get vector
        df = df[(df.type=='vector') & (df.name==statistic+':vector')]
        data_per_run = []
        if strategy == 'means':
            means = [np.mean(row['vecvalue']) for _,row in df.iterrows()]
            data_per_run.append({
                'endog': means
            })
        elif strategy == 'weighted_means':
            weighted_means = []
            for _, row in df.iterrows():
                time = row['vectime']
                value = row['vecvalue']
                weight = time[1:] - time[:-1]
                amount = value[:-1]
                weighted_mean = np.mean(weight*amount)
                weighted_means.append(weighted_mean)
            data_per_run.append({
                'endog': weighted_means
            })
        elif strategy == 'batch_values':
            time = df['vectime'].iloc[0]
            value = df['vecvalue'].iloc[0]
            timedist = math.floor( (time[-1] - time[0])/nbatches )
            t0 = time[0]
            for i in range(nbatches):
                data_per_run.append({
                    'endog': value[(t0+i*timedist < time) & (time < t0+(i+1)*timedist)]
                })
        elif strategy == 'equi_timedist':
            for _, row in df.iterrows():
                time = row['vectime']
                value = row['vecvalue']
                timedist = math.floor( (time[-1] - time[0])/nbatches )
                t0 = time[0]
                batch_data = [value[time < t0+(i+1)*timedist][-1] for i in range(nbatches)]
                data_per_run.append({
                    'endog': batch_data
                })
        elif strategy == 'batch_means':
            for _, row in df.iterrows():
                time = row['vectime']
                value = row['vecvalue']
                timedist = math.floor( (time[-1] - time[0])/nbatches )
                t0 = time[0]
                batch_means = [np.mean(value[(t0+i*timedist < time) & (time < t0+(i+1)*timedist)]) for i in range(nbatches)]
                data_per_run.append({
                    'endog': batch_means
                })
        elif strategy == 'weighted_batch_means':
            for _, row in df.iterrows():
                time = row['vectime']
                value = row['vecvalue']
                timedist = math.floor( (time[-1] - time[0])/nbatches )
                t0 = time[0]
                weighted_batch_means = []
                for i in range(nbatches):
                    batch_times = time[(t0+i*timedist < time) & (time < t0+(i+1)*timedist)]
                    batch_value = value[(t0+i*timedist < time) & (time < t0+(i+1)*timedist)]
                    weight = batch_times[1:] - batch_times[:-1]
                    amount = batch_value[:-1]
                    weighted_mean = np.mean(weight*amount)
                    weighted_batch_means.append(weighted_mean)
                data_per_run.append({
                    'endog': weighted_batch_means
                })
        else:
            for _, row in df.iterrows():
                # if hasattr(self.model_class, 'model_class') and issubclass(self.model_class.distribution, rv_discrete):
                #     value = value.astype(int)
                data_per_run.append({
                    'endog': row['vecvalue']
                })
        self.data_per_run = np.array(data_per_run)
        
    def generate_bs_dataset(self, nrep, parametric):
        print('###############################################################')
        print('### Generating bootstrap data #################################')
        print('###############################################################')
        ### fit models for and bootstrap data each run 
        fitresults_per_run = []
        last_10_elapsed_times = []
        for i, data in enumerate(self.data_per_run):
            t = time.process_time()
            ####################################################
            with self.suppress_stdout():
                endog = data['endog']
                if 'exog' in data:
                    exog = data['exog']  
                else:
                    exog = np.zeros_like(endog)
                tmp_model = self.model_class(endog, exog, 
                                             start_params=self.theta_0)
                tmp_results = tmp_model.fit()
                tmp_results.bootstrap_mle(nrep=nrep, parametric=parametric) # 1000
                dict = tmp_results.get_results_as_dict()
                fitresults_per_run.append(dict)
            ####################################################
            # estimat time to go
            elapsed_time = time.process_time() - t
            last_10_elapsed_times.append(elapsed_time)
            if i >= 10: last_10_elapsed_times.pop(0)
            min_to_go = np.round( sum(last_10_elapsed_times)*(len(self.data_per_run) - i)/
                                    (60*len(last_10_elapsed_times) ), 2)
            print('now at {}/{} - {}min to go - {}s per iteration'.format(i, 
                    len(self.data_per_run), min_to_go, elapsed_time))
        self.save_dict_to_file(self.bootstrapcb_dataset, fitresults_per_run) 
    
    def init_dataset(self, show_first_run=False):
        ### get dataset from file
        self.fitresults_per_run = self.get_dict_from_file(self.bootstrapcb_dataset) # TODO assert
        # init results with some data
        endog = np.array(self.fitresults_per_run[0]['endog'])
        exog = np.array(self.fitresults_per_run[0]['exog'])
        model = self.model_class(endog=endog, exog=exog, start_params=self.theta_0)
        self.results = model.fit() # TODO dont have to fit here
        
        if show_first_run:
            self.show_dataset_at_run(0)

    def show_dataset_at_run(self, i, show_seperated=False, save_figures=False, nx=40,
                            cr_xmin=None, cr_xmax=None, cr_ymin=None, cr_ymax=None,
                            cb_xmin=None, cb_xmax=None, cb_ymin=0, cb_ymax=1):  
        self.results.set_results_from_dict(self.fitresults_per_run[i])      
        
        self.data_plots(nx=nx, xmin=cb_xmin, xmax=cb_xmax)
        if show_seperated and not save_figures: plt.show()

        if len(self.theta_0) == 2: 
            self.confregion_plots(xmin=cr_xmin, xmax=cr_xmax, ymin=cr_ymin, ymax=cr_ymax)
            if show_seperated and not save_figures: plt.show()

        self.confband_plots(nx=nx, xmin=cb_xmin, xmax=cb_xmax, ymin=cb_ymin, ymax=cb_ymax)
        if not save_figures:
            plt.show()
        else:
            self.save_figures()

    def simstudy(self, xs, tests, configs, levels, which, show, save=True):
        txt = ''
        if which=='confregion' or which=='both':
            line = ( '###############################################################\n' +
                     '### Confregion results ########################################\n' +
                     '###############################################################\n')
            print(line)
            simstudy_cr_results = self.sim_test_confregion(tests, levels, show=show)
            txt += line
            for item in simstudy_cr_results:
                line = '{}:\n {}\n'.format(item, simstudy_cr_results[item])
                txt += line
                print(line)
        if which=='confband' or which=='both':
            line = ( '###############################################################\n' +
                     '### Confband results ##########################################\n' +
                     '###############################################################\n')
            print(line)
            simstudy_cb_results = self.sim_test_confband(xs, configs, levels, show=show)
            txt += line
            for item in simstudy_cb_results:
                line = '{}:\n {}\n'.format(item, simstudy_cb_results[item])
                txt += line
                print(line)
        if save:
            file = open(self.simstudy_dataset, "w")
            file.write(txt)
            file.close() 
    
    def fmt(self, x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"     
        
    def plot_theta(self, plot): 
        theta_0 = self.theta_0
        plot.plot(theta_0[0], theta_0[1], 'o', label='true value')

    def plot_confregion(self, plot, statistic, level):
        mle = self.results.params
        cov = self.results.cov_params()
        xmin = mle[0] - 5*np.sqrt(cov[0][0])
        xmax = mle[0] + 5*np.sqrt(cov[0][0])
        ymin = mle[1] - 5*np.sqrt(cov[1][1])
        ymax = mle[1] + 5*np.sqrt(cov[1][1])
        xs   = np.linspace(xmin, xmax, 50)
        ys   = np.linspace(ymin, ymax, 50)
        X, Y = np.meshgrid(xs, ys)
        XY   = np.dstack((X, Y))
        Z = statistic(XY)
        plot.contour(X, Y, Z, levels=[level], colors='grey', linestyles='--')  
        
    def sim_test_confregion(self, tests, levels, show=False):
        # if show
        if show:
            fig = self.get_new_fig()
            confregion_plot = fig.add_subplot(111)
            self.plot_theta(confregion_plot)
        # do statistics for each simulated run
        robj = []
        last_10_elapsed_times = []
        for i, fitrestult in enumerate(self.fitresults_per_run):
            t = time.process_time()
            ####################################################
            with self.suppress_stdout():
                # init result class with fitresults from run
                self.results.set_results_from_dict(fitrestult)
                # confidence regions
                results_per_run = {}
                for level in levels:
                    for test in tests:
                        config_name = 'coverage_{}_{}'.format(level, test)
                        if test=='w':
                            t0 = self.results.wald_statistic(self.theta_0)[0]
                            t_hat = chi2.ppf(1 - level, df=self.df)
                        if test=='w_bs':
                            t0 = self.results.wald_statistic(self.theta_0)[0]
                            T = self.results.wald_statistic(self.results.Params_bs)
                            t_hat = np.quantile(T, 1 - level)
                        if test=='studw_bs':
                            t0 = self.results.wald_statistic(self.theta_0)[0]
                            T = self.results.wald_statistic(self.results.Params_bs, self.results.Normalized_cov_params_bs)
                            t_hat = np.quantile(T, 1 - level)
                        if test=='lr':
                            t0 = self.results.wald_statistic(self.theta_0)[0]
                            t_hat = chi2.ppf(1 - level, df=self.df)
                        if test=='lr_bs':
                            t0 = self.results.likelihood_ratio_statistic(self.theta_0)[0]
                            T = self.results.likelihood_ratio_statistic(self.results.Params_bs)
                            t_hat = np.quantile(T, 1 - level)
                        results_per_run[config_name] = [t0, t_hat]
                # record results from run
                robj.append(results_per_run)
                # if show
                if show:
                    self.plot_confregion(confregion_plot, self.results.wald_statistic, chi2.ppf(0.9, df=self.df))
            ####################################################
            # estimat time to go
            elapsed_time = time.process_time() - t
            last_10_elapsed_times.append(elapsed_time)
            if i >= 10: last_10_elapsed_times.pop(0)
            min_to_go = np.round( sum(last_10_elapsed_times)*(len(self.fitresults_per_run) - i)/
                                  (60*len(last_10_elapsed_times) ), 2)
            print('now at {}/{} - {}min to go - {}s per iteration'.format(i, 
                   len(self.fitresults_per_run), min_to_go, elapsed_time))
        robj = np.array(robj) 
        # if show
        if show:
            plt.show()
        # coverages for each run
        simstudy_results = {}
        for level in levels:
            for test in tests:
                config_name = 'coverage_{}_{}'.format(level, test)
                t0 = np.array([r[config_name][0] for r in robj])
                t = np.array([r[config_name][1] for r in robj])
                successfull_runs = (t0 < t)
                simstudy_results[config_name] = len(robj[successfull_runs])/len(robj)
        return simstudy_results
    
    def sim_test_confband(self, xs, configs, levels, show=False):
        # do statistics for each simulated run
        robj = []
        last_10_elapsed_times = []
        for i, fitrestult in enumerate(self.fitresults_per_run):
            t = time.process_time()
            ####################################################
            with self.suppress_stdout():
                # init result class with fitresults from run
                self.results.set_results_from_dict(fitrestult)
                # confidence bands
                results_per_run = {}
                for level in levels:
                    for config in configs:
                        config_name = 'cb_{}_{}_{}'.format(level, config['test'], config['method'])
                        results_per_run[config_name] = self.results.confband(xs, alpha=level, test=config['test'], method=config['method'])
                # record results from run
                robj.append(results_per_run)
            ####################################################
            # estimat time to go
            elapsed_time = time.process_time() - t
            last_10_elapsed_times.append(elapsed_time)
            if i >= 10: last_10_elapsed_times.pop(0)
            min_to_go = np.round( sum(last_10_elapsed_times)*(len(self.fitresults_per_run) - i)/
                                  (60*len(last_10_elapsed_times) ), 2)
            print('now at {}/{} - {}min to go - {}s per iteration'.format(i, 
                   len(self.fitresults_per_run), min_to_go, elapsed_time))
        robj = np.array(robj) 
        # coverages for each run
        y_0 = self.results.mle(xs)
        simstudy_results = {}
        for level in levels:
            for config in configs:
                config_name = 'cb_{}_{}_{}'.format(level, config['test'], config['method'])
                y_l = [r[config_name][:,0] for r in robj]
                y_u = [r[config_name][:,1] for r in robj]
                succesfull_runs = ( np.array([all(y_l[i] < y_0) for i,_ in enumerate(self.fitresults_per_run)]) & 
                                    np.array([all(y_0 < y_u[i]) for i,_ in enumerate(self.fitresults_per_run)]) )
                simstudy_results[config_name] = len(robj[succesfull_runs])/len(robj)
        return simstudy_results
    