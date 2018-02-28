# -*- coding: utf-8 -*-
import numpy as np
import math
#import matplotlib.pyplot as plt


def calc_exceedance_freqs_exposures(limit100, excess100, cedant_premium, 
                                    loss_ratio, prog_ID, exponent, points):
    '''
    exponent - b (power parameters) [np.array]
    points   - x value points, enter 'default' to use auto scale [np.array]
    prog_ID  - currently not used, will be used for stacking [list of str]
    limit100 - 100% limits
    excess100 - 100% excess
    '''

    i1 = np.array( excess100 )
    i2 = i1 + np.array( limit100 )    
    cedant_loss = cedant_premium * loss_ratio
    
    max_i2 = max(i2)
    min_i1 = min(i1)
    
    c_i1 = i1**exponent
    c_i2 = i2**exponent
    
    if isinstance(points, str):    
        min_i1 = 1e5
        x = np.linspace(min_i1, max_i2-1, 100)
        x = np.append(x, max_i2)
        x = np.append(x, max_i2*1.02)
    else:
        x = np.array( points )
        
    f = []
    
    n = len(x)
    
    for i in range( n ):
        x1 = x[i]
        x2 = x1+1
        
        c_r1 = x1**exponent
        c_r2 = x2**exponent
        
        re_fract = (c_r2-c_r1)/(c_i2-c_i1)
        re_loss  = re_fract * cedant_loss
        
        re_loss[ x1>=i2 ] = 0.0
        
        freq = sum(re_loss)    
        f.append( freq )
    x = x.tolist()    
    return x, f


def layer_claims(claims, limit, excess):
    x = claims - excess
    x[x<0] = 0
    x[x>limit] = limit
    return x
    
def aggregate_claims(claims, claims_per_year):
    zc_cm = np.concatenate( ( np.array([0.0]),  np.cumsum( claims ) )  )
    zf_cm = np.cumsum( claims_per_year )
    
    a = zc_cm[zf_cm]
    
    agg = np.diff( np.concatenate(  (np.array([0]), a),   axis=0) )
    return agg
    
def pareto_alpha_beta_from_frequencies(entry_freq, exit_freq, severity_lower, severity_upper):
    '''
    Usage: \n alpha, beta, mean = pareto_alpha_beta_from_frequencies(entry_freq, exit_freq, severity_lower, severity_upper)
    
    - severity_lower is your excess point of the modelled layer
    - severity_upper is the upper point of the modelled layer (excess + limit)
    
    '''
    
    e1 = severity_lower
    e2 = severity_upper
    f1 = entry_freq
    f2 = exit_freq
    ln = math.log
    alpha = -( ln(f2) - ln(f1)  ) / (  ln(e2) - ln(e1)  )
    
    beta = np.exp(    ln(f1)/alpha + ln(e1)    )
    
    # The calcs below are the integral of expected value between the bounds
    t1 = 1.0 / (1.0 - alpha)
    t2 = alpha*beta**(alpha)
    t3 = severity_lower**(-1.0*alpha + 1)
    t4 = severity_upper**(-1.0*alpha + 1)
    mean1 = t1*t2*( t4  - t3)
    mean2 = severity_lower*(beta**alpha)*( severity_upper**(-alpha) -  severity_lower**(-alpha)  )
    mean3 = exit_freq * (severity_upper-severity_lower)
    mean = mean1 + mean2 + mean3
    cc = convert_to_money
    print('alpha: ' + str(alpha) + ', beta: ' + str(beta) + ', mean: ' + str(mean)  )
    msg = 'x1: {}, x2: {}, alpha: {}, beta: {}, mean: {}'.format(
            cc(e1), cc(e2), alpha, beta, mean)
    print(msg)
    return alpha, beta, mean
    
def random_poisson_numbers(mean, num_simulations):
    poisson_numbers = np.random.poisson(mean, num_simulations)
    return poisson_numbers
    
def convert_to_money(value):
    string = '{:,.0f}'.format(value)
    return string
    
def convert_to_2dp_string(value):
    x = "%.2f" % value
    return x
    
def convert_to_3dp_string(value):
    x = "%.3f" % value
    return x
    
def convert_to_4dp_string(value):
    x = "%.4f" % value
    return x

def convert_to_5dp_string(value):
    x = "%.5f" % value
    return x

def convert_to_6dp_string(value):
    x = "%.6f" % value
    return x

def convert_to_7dp_string(value):
    x = "%.7f" % value
    return x
    
def convert_to_14dp_string(value):
    x = "%.14f" % value
    return x    
    
def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
    
class Pareto:
    
    
    def __init__(self, limit, excess, frequency):
        '''
        Usage:         
        
        par = act.Pareto(limit, excess, frequency)

        par = act.Pareto([5e5, 1e6, 2e6], [5e5, 1e6, 2e6], [1.0, 0.51, 0.2, 0.005])
        
        frequency must have a final frequency which makes the exit freq for  last layer
        '''
        num_excess    = len(excess)
        num_limit     = len(limit)
        num_frequency = len(frequency)
        
        if num_excess != num_limit:
            raise ValueError('You entered ' + str(num_excess) +
                             ' excess points and ' + str(num_limit) +
                             ' limit points, there MUST be equal number of each')
                             
        if num_excess != (num_frequency-1):
            raise ValueError('You need to enter one more frequency than number of exit points, which acts as the final exit frequeny')
        
        self.limit       = None
        self.excess      = None
        self.frequency   = None
        self.num_layers  = None
        self.alpha       = []
        self.beta        = []
        self.layer_means = []
        self.analytical_mean = 0
        self.debug_print     = True
        self.base_frequency  = None
        self.frequency_distribution = 'poisson'
        self.sim_frequencies  = None
        self.sim_indiv_claims = None        
        self.num_simulations  = 10
        self.limit      = np.array( limit )
        self.excess     = np.array( excess )
        self.frequency  = np.array( frequency )
        self.num_layers = num_excess
        self.base_frequency = frequency[0]
        
        self.calculate_alpha_beta()
        
    def calculate_alpha_beta(self):
        
        for i in range( self.num_layers ):
            severity_lower = self.excess[i]
            severity_upper = severity_lower + self.limit[i]
            entry_freq     = self.frequency[i]   / self.base_frequency # Normalise as sev dist
            exit_freq      = self.frequency[i+1] / self.base_frequency
            
            alpha, beta, mean = pareto_alpha_beta_from_frequencies(entry_freq, 
                                                                   exit_freq, 
                                                                   severity_lower, 
                                                                   severity_upper)
            self.layer_means.append( mean )                                                       
            self.analytical_mean += mean
            self.alpha.append( alpha )                                                                   
            self.beta.append( beta )            
        
        #self.printt('Section analytical mean: ' + convert_to_money(self.analytical_mean) )
            
    def calculate_claims(self):
        self.generate_frequencies()
        self.generate_severities()
        #self.sim_agg_claims = aggregate_claims(self.sim_indiv_claims, self.sim_frequencies)
        
    def generate_frequencies(self):
        if self.frequency_distribution == 'poisson':
            self.sim_frequencies = random_poisson_numbers(self.base_frequency, self.num_simulations)
        else:
            raise ValueError('Invalid frequency distribution stated')
            
        #self.print('Mean of freqs: ' + str(self.sim_frequencies.mean() )) 
        
    def generate_severities(self):
        num_claims = self.sim_frequencies.sum()
        num_paretos   = len(self.alpha)
        uniform_rands = np.random.uniform(0,1,num_claims)
        indiv_claims  = np.zeros( (num_claims) )
        
        ln = np.log
        
        idx_min_to_max = uniform_rands.argsort()
        idx_max_to_min = uniform_rands.argsort()[::-1]
        r_sorted = uniform_rands[idx_max_to_min]
                
        for i in range(num_paretos):
            y1 = self.frequency[i]   / self.base_frequency
            y2 = self.frequency[i+1] / self.base_frequency
            
            lower_idx = find_nearest_index(r_sorted, y1)
            upper_idx = find_nearest_index(r_sorted, y2)
            #print('Lower ind: ' + str(lower_idx) + ', Upper: ' + str(upper_idx) )
            
            rands = r_sorted[lower_idx : upper_idx]            
            
            alpha = self.alpha[i]
            beta  = self.beta[i]
            
            temp_claims = np.exp(  ln(beta) - ln(rands)/alpha)
            
            indiv_claims[lower_idx : upper_idx] = temp_claims
            
            if i == num_paretos-1:
                final_upper_severity_point = self.excess[-1] + self.limit[-1]
                indiv_claims[ upper_idx:,] = final_upper_severity_point
            
        indiv_claims2 = indiv_claims*0 
        indiv_claims2[idx_max_to_min] = indiv_claims
        self.sim_indiv_claims = indiv_claims2
        test = indiv_claims2[indiv_claims2 < final_upper_severity_point ]  
        
        #print( test.mean() )               
        
        
    def printt(self, msg):
        if self.debug_print == True:
            print(msg)
        
    
            
class Section:
    
    def __init__(self, section_type):
        self.section_type = section_type
        self.section_type = None
        self.mean_claims  = None
        self.claim_engine = None
        self.num_simulations = 0
    def set_num_simulations(self, num_simulations):
        self.num_simulations = num_simulations
    def set_mean_number_claims(self, mean):
        self.mean_claims = mean
        
    def set_Pareto_parameters(self, limit, excess, frequency):
        self.claim_engine = Pareto(limit, excess, frequency)
        self.claim_engine.num_simulations = self.num_simulations
        self.claim_engine.calculate_claims()
        
    def get_indiv_claims(self):
        indiv_claims = self.claim_engine.sim_indiv_claims
        return indiv_claims
        
    def get_frequencies(self):
        frequencies = self.claim_engine.sim_frequencies
        return frequencies
        
     
class Simulation:
    
    def __init__(self, num_simulations):        
        
        self.sections = []
        self.layers   = []
        self.num_layers = 0
        self.debug_print = True
        self.num_simulations = num_simulations
        self.coverage = None   
        self.results = {}
        #self.print('Simulation object created')
        
    def add_section(self, section):
        
        section.set_num_simulations( self.num_simulations )
        self.sections.append( section )
        
    def add_layer(self, layer):
        if self.coverage is not None:
            raise ValueError('Cannot add a layer after coverage it set, add layers first')
        
        self.num_layers += 1
        
        self.layers.append( layer )
        
        
    def set_coverage(self, coverage): 
        
        num_sections = len(self.sections)
        num_layers   = len(self.layers)
        
        for c in coverage:
            if len(c) != num_sections:
                raise ValueError('At least one of your coverages is not right length, i.e. not the same as number of sections')
                
        if len(coverage) != num_layers:
            raise ValueError('Your covereage map does not have as same num rows as num sections')
        
        self.coverage = coverage
        
    def run_claims_through_structure(self):
        num_layers = len(self.layers)

        for i in range(num_layers):
            self.printt('           LAYER {} now processing ----------------------'.format(i+1) )
            self.process_layer_claims(i) 
            self.process_agg_claims(i)            
            
        for layer in self.layers:
            # this loop is only for diagnostics, doesnt do anything
            results = layer.results            
            
            
    def run_percentiles(self):
        prog_claims = np.zeros( self.num_simulations )
        prog_profit = np.zeros( self.num_simulations )
        increment = 0.1
        perc_points = np.arange(0,  ( 100+increment) ,   increment)
        #perc_points_reverse = np.flip(perc_points, 0)
        perc_points_reverse = 100-perc_points
        perc_points_reverse = perc_points 
        
        
        for layer in self.layers:
            results = layer.results
            claims = results['agg_claims_layered']
            claims_cdf = np.percentile(claims, perc_points)
            results['agg_claims_cdf_x'] = claims_cdf
            results['agg_claims_cdf_y'] = perc_points
            
            profit = results['profit']
            profit_cdf = np.percentile(profit, perc_points_reverse)
            results['profit_cdf_x'] = profit_cdf
            results['profit_cdf_y'] = perc_points_reverse
                        
            prog_claims += claims
            prog_profit += profit
            
            if False:
                plt.figure()
                plt.plot( claims_cdf, perc_points )
                plt.figure()
                plt.plot( x, hist )
        
        results = self.results
        claims_cdf = np.percentile(prog_claims, perc_points)
        results['agg_claims_cdf_x'] = claims_cdf
        results['agg_claims_cdf_y'] = perc_points        
        
        profit_cdf = np.percentile(prog_profit, perc_points_reverse)
        results['profit_cdf_x'] = profit_cdf
        results['profit_cdf_y'] = perc_points_reverse
        
            
            
    def process_layer_claims(self, layer_number):
        layer = self.layers[layer_number]
        coverage = self.coverage[layer_number]    
        layer.results['indiv_claims_layered'] = []
        section_counter = -1
        
        for section in self.sections:
            section_counter += 1
            #self.printt(' -- Section {}'.format(section_counter+1) )
            temp_dict = {}
            if coverage[section_counter] == True:
                #self.printt('      --Layer covered')
                indiv_claims = section.get_indiv_claims()
                limit  = layer.indiv_limit[section_counter]
                excess = layer.indiv_excess[section_counter]
                
                indiv_claims_layered = layer_claims(indiv_claims, limit, excess)

                temp_dict['indiv_claims_layered'] = indiv_claims_layered
                temp_dict['frequencies']          = section.get_frequencies()
                temp_dict['section_covered']      = True                
                #self.printt('         > Mean indiv raw claims    : ' + convert_to_money(indiv_claims.mean()) ) 
                #self.printt('         > Mean indiv layered claims: ' + convert_to_money(indiv_claims_layered.mean()) ) 
                
            else: 
                #self.printt('      --Layer Not covered')
                temp_dict['section_covered']      = False
            layer.results['indiv_claims_layered'].append( temp_dict )

    def process_agg_claims(self, layer_number):
        
        layer = self.layers[layer_number]
        total_agg_claims = np.zeros( self.num_simulations )
        layered_indiv_claims = layer.results['indiv_claims_layered']
        
        for layered_claims in layered_indiv_claims:
            if layered_claims['section_covered'] == True:
                indiv_claims = layered_claims['indiv_claims_layered']
                frequencies  = layered_claims['frequencies']
                
                agg_claims = aggregate_claims(indiv_claims, frequencies)
                layered_claims['agg_claims'] = agg_claims
                
                total_agg_claims += agg_claims
                
        agg_limit = layer.agg_limit
        agg_deduct = layer.agg_deduct
        agg_claims_layered = layer_claims(total_agg_claims, agg_limit, agg_deduct)
        
        layer.results['agg_claims_unlayered'] = total_agg_claims        
        layer.results['agg_claims_layered']   = agg_claims_layered
        #self.printt('   --Mean agg unlayered claims: ' + convert_to_money(total_agg_claims.mean()) )
        #self.printt('   --Mean agg layered claims  : '   + convert_to_money(agg_claims_layered.mean()) )
                
#        layer.agg_claims_unlayered = total_agg_claims
#        layer.agg_claims_layered   = agg_claims_layered
#        
#        mean = layer.agg_claims_layered.mean()
#        regular_std = layer.agg_claims_layered.std()
#        msg = 'Mean of layered agg claims:    ' + convert_to_money( mean )
#        self.printt(msg)        
#        msg = 'Reg std of layered agg claims: ' + convert_to_money( regular_std )
#        self.printt(msg)        
        
        #x = agg_claims_layered[ agg_claims_layered < 0]
        #msg = 'Downside std of layered agg claims: ' + convert_to_money( x.std() )                
        #self.print(msg)                
            
            
    def run_profit_loss(self):
        num_layers = len(self.layers)

        for i in range(num_layers):
            self.run_profit_loss_for_layer(i)
            
            
    def run_profit_loss_for_layer(self, layer_number):
        layer = self.layers[layer_number]
        pricing = layer.pricing
        results = layer.results
        
        agg_claims = results['agg_claims_layered']
        
        premium           = pricing['upfront_premium_money']
        upfront_brokerage = pricing['upfront_brokerage_percentage']
        expense_ratio     = pricing['expense_percentage']
        internal_expenses = premium * ( upfront_brokerage + expense_ratio)
        premium_after_aqc_costs = premium * (1 - upfront_brokerage - expense_ratio)

        profit  = premium_after_aqc_costs - agg_claims
        
        profit_mean = profit.mean()
        profit_std  = profit.std()
        
        agg_claims_mean = agg_claims.mean()
        loss_ratio      = agg_claims_mean / premium
        combined_ratio  = (agg_claims_mean+internal_expenses)/premium
        
        results['expense_internal_money'] = expense_ratio*premium
        results['premium_money'] = premium
        results['upfront_brokerage_money'] = upfront_brokerage*premium
        results['profit'] = profit
        results['loss_ratio'] = loss_ratio
        results['combined_ratio'] = combined_ratio
        results['expected_loss_money'] = agg_claims_mean
        msg = '''Premium={}, E[loss]={}, LR={}, CR={}'''.format(
                convert_to_money(premium), convert_to_money(agg_claims_mean), 
                convert_to_2dp_string(100*loss_ratio), convert_to_2dp_string(100*combined_ratio)  )
        #print(msg)

    def run_capital_calcs(self):
        profit_neg = profit[profit < premium*.50]
        if len(profit_neg) < 2:
            profit_neg_std = 0.0
        else:
            profit_neg_std = profit_neg.std()
            profit_neg_std = np.sqrt( np.sum( np.power(profit_neg,2)  )/len(profit_neg)  )  
        
        self.printt('Loss ratio: ' + convert_to_2dp_string(100*loss_ratio) + '%' )
        self.printt('Comb ratio: ' + convert_to_2dp_string(100*combined_ratio) + '%' )
        self.printt('Profit mean undisc: ' + convert_to_money( profit_mean) )
        roc_normal = 100*profit_mean/profit_std
        self.printt('Profit all std:     ' + convert_to_money( profit_std) + ', ROC: ' + convert_to_2dp_string(roc_normal) + '%' )
        roc2 = 100*profit_mean/profit_neg_std
        self.printt('Profit neg std:     ' + convert_to_money( profit_neg_std) + ', ROC: ' + convert_to_2dp_string(roc2) + '%' )
        
        
        
    def printt(self, msg):
        if self.debug_print == True:
            print(msg)
        
class Layer:
    def __init__(self):        
        self.layer_num       = None
        self.debug_print     = True
        self.indiv_limit     = None
        self.indiv_excess    = None
        self.agg_limit       = None
        self.agg_deduct      = None
        self.indiv_claims_layered = []
        self.agg_claims_unlayered = []
        self.agg_claims_layered   = []
        self.results         = {}
        self.pricing         = None
        self.set_pricing_dictionary()
        
        #self.print('Layer object created')
        
    def set_pricing_dictionary(self):
        pricing = {}
        pricing['upfront_brokerage_percentage'] = 0.0
        pricing['upfront_premium_money']        = 0.0        
        pricing['expense_percentage']           = 0.0                
        pricing['ceding_commission_percentage'] = 0.0
        self.pricing = pricing
        
    def set_agg_limit_deductible(self,limit=9.9999e20, deductible=0):
        self.agg_limit = limit
        self.agg_deduct = deductible
        
    def set_indiv_limit_excess(self, limit, excess):
        self.indiv_limit  = limit
        self.indiv_excess = excess
        
    def set_upfront_brokerage(self, upfront_brokerage):
        self.pricing['upfront_brokerage_percentage'] = upfront_brokerage
        
    def set_upfront_premuim(self, upfront_premium):
        self.pricing['upfront_premium_money'] = upfront_premium
        
    def set_expense_ratio(self, expense_ratio):
        self.pricing['expense_percentage'] = expense_ratio
        
    def set_ceding_commission(self, ceding_commission):
        self.pricing['ceding_commission_percentage'] = ceding_commission
        
        
    def printt(self, msg):
        if self.debug_print == True:
            print(msg)
        
        
        
            
                       
        







































