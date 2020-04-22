#!/usr/bin/env python
# coding: utf-8

# The code in this notebook will serve as the basis for the CSCI 3352 final project for spring 2020. 
# 
# **Name:** Nicholas Palmer 
# 
# **Date:** 1/4/2020
# 
# The purpose of this project will be to investigate how graph structure affects the spread of a contagion through a population. The idea with this type of project is to see what strategies for modifying graph structure can be used to "flatten the curve" effectively so that the health care system will not be overwhelmed by the pandemic. 

# In[5]:


import networkx as nx
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pylab as plt
import copy
import time

print("Script running now")

plt.rcParams['figure.figsize'] = 15,10
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


# In[6]:


#Preproccess the file that was provided
#This code should only be run once
edge_file = open('data/out.contact', 'r')
file_lines = edge_file.readlines()

#The file to store the processed data
processed_file = open('data/contacts.txt', 'w+')

to_nodes = []
from_nodes = []
timestamps = [] #Storing these to change from UNIX timestamps to simulation timestamps

for line in file_lines:
    line_arr = line.split()
    new_line = str(line_arr[0]) + ' ' + str(line_arr[1]) + ' ' + str(line_arr[3]) + '\n'
    processed_file.write(new_line)
    
processed_file.close()
edge_file.close()


# In[7]:


#Read the file in line by line. Add contents to dictionary
#This approach was chosen because there were extraneous data in the edge list
#Additionally, the list of times needed to be preserved
edge_list = open('data/contacts.txt', 'r')
edges_lines = edge_list.readlines()

#The edges in the graph from the proximity data
edges = []
simple_edges = []
#Store each interaction in the dataset
for line in edges_lines:
    edge_values = line.split()
    edge = (int(edge_values[1]), int(edge_values[0]), 1)
    simple_edge = (int(edge_values[1]), int(edge_values[0]))
    edges.append(edge)
    simple_edges.append(simple_edge)
    
#Remove duplicate edges for an undirected graph
simple_edges = list(dict.fromkeys(simple_edges))


# In[8]:

#Generate the interaction network
interaction_network = nx.Graph(simple_edges)


# In[40]:


#Draw graphs with a labelling
#Used during simulation
def drawGz(G,z):
    colors = {'S':'#99FF99', 'A':'#FFB570', 'Y':'#FF99CC', 'E':'#FF3333', 'R':'#6666FF'}

    node_colors = []
    for i in G.nodes():
        node_colors.append(colors[z[i]])
    nsize  = 600
    flabel = True

    if G.order() > 50:
        nsize  = 100
        flabel = False
        
    nx.draw_networkx(G,with_labels=flabel,node_size=nsize,width=2,node_color=node_colors) # draw it pretty
    limits=plt.axis('off')                                      # turn off axes
    plt.show() 

    return

#Draw graphs without a labelling
#Useful for testing new types of graphs before simulating
def draw_G(G):
    print(f'graph with n={G.order()} nodes and m={G.size()} edges:')
    limits=plt.axis('off') 
    nx.draw_networkx(G,with_labels=True,node_size=600,width=2) # draw it pretty                                    # turn off axes
    plt.show()
    return

#Generate a healty population labeling. All nodes susceptible
def generateZt(G, initial_infected, At_initial, Yt_initial):
    zt = {}
    for node in G.nodes():
        zt[node] = 'S' #Initially make all nodes susceptible
        
    initial_keys = list(np.random.choice(list(zt.keys()), initial_infected, replace=False))
    for key in initial_keys[0:At_initial]:
        zt[key] = 'A'
    
    for key in initial_keys[At_initial:At_initial+Yt_initial]:
        zt[key] = 'Y'
    
    for key in initial_keys[At_initial+Yt_initial:]:
        zt[key] = 'E'    
    
    return zt

#Check the beta and gamma values for each infected group to ensure they are reasonable
#Also check for the average R_0 value to ensure it is reasonable
def checkBetaAndGamma(group_coefficients):
    r0s = []
    r0s.append(group_coefficients['A'][0]/group_coefficients['A'][1])
    r0s.append(group_coefficients['Y'][0]/group_coefficients['Y'][1])
    r0s.append(group_coefficients['E'][0]/group_coefficients['E'][1])
    average_r0 = np.mean(r0s)
    
    print("R0 for the asymptomatic group is " + str(r0s[0]))
    print("R0 for the symptomatic group is " + str(r0s[1]))
    print("R0 for the severe group is " + str(r0s[2]))
    print("The average R0 for this infection is " + str(average_r0))

#Implement simple social distancing on the graph G
#Delete each edge with probability alpha
def socialDistance(G, alpha):
    if(alpha == 0):
        return G
    for u, v in G.edges():
        #Remove with probability alpha
        if(np.random.rand(1)[0] < alpha):
            G.remove_edge(u, v)
    return G
    
#Description: 
#Runs a parameterized simulation of the SASER model on a given network
#Parameters:
#G: The network which the simulation runs on
#initial_infected: The number of infected individuals at t=1. The first individual is chosen to be symptomatic. The rest are chosen randomly
#severity_pdf: The probability density function for each level of severity when an individual becomes infected
#group_coefficients: A dictionary containing tuples for beta and gamma values for each of the A, Y, and E groups
#global_p: The global mixture probability. The probability that individuals interact with a random, non-adjacent node
#percent_infected: The percent of the population that must be infected before implementing social distancing
#alpha: Probability of deleting each edge in the network after social distancing is implemented
#verbose: A flag for verbose printing. Not recommended for large graphs
#draw: Boolean flag for drawing the network at each step of the simulation. Not recommended for large graphs
def runSAYER(G, initial_infected, severity_pdf, group_coefficients, global_p, percent_infected, alpha, verbose=False, draw=False):
    #Make sure that G is a deep copy so as not to change the original
    G = copy.deepcopy(G)
                
    #Optionally print some information about R0
    if(verbose):
        checkBetaAndGamma(group_coefficients)
    
    St = [] # S(t), time series of number of Susceptible nodes per time step t
    At = [] # A(t), time series of number of Asymptomatic nodes per time step t
    Yt = [] # S(t), time series of number of sYmptomatic nodes per time step t
    Et = [] # E(t), time series of number of sEvere nodes per time step t
    Rt = [] # R(t), time series of number of R nodes per time step t
    
    It = [] # I(t), time series of number of Infected nodes per time step t (total)
    t  = 1 #Simulation clock
    
    infected_groups = list(group_coefficients.keys()) #Possible labels for infected individuals
    
    #Choose which groups the initial infected individuals fall into
    patient_zeros = list(np.random.choice(infected_groups, initial_infected, severity_pdf))
    At_initial = patient_zeros.count('A')
    Yt_initial = patient_zeros.count('Y')
    Et_initial = patient_zeros.count('E')
    
    #Initialize timeseries arrays with the proper number of individuals
    St.append(len(G.nodes()) - initial_infected)
    At.append(At_initial)
    Yt.append(Yt_initial)
    Et.append(Et_initial)
    Rt.append(0)
    
    It.append(initial_infected)
    
    #Optionally print some information about the first infected individuals
    if(verbose):
        print("Count of patient zero(s):")
        print("Asymptomatic: " + str(At_initial))
        print("Symptomatic: " + str(Yt_initial))
        print("Severe: " + str(Et_initial))
    
    #Initialize the labelling with patient zeroes
    #This method was chosen to reduce computation time in setting labels from a group of 3
    zt = generateZt(G, initial_infected, At_initial, Yt_initial)
    
    #Optionally draw the graph at the first time step
    if(draw):
        print(f'time step {t}')
        drawGz(G,zt)
    distancingFlag = False
    #Run the simulation while there are any people in infected groups
    while any(infected_group in zt.values() for infected_group in infected_groups):
        #Check if social distancing will be implemented
        if(distancingFlag == False):
            if(It[t-1]/len(G.nodes()) >= percent_infected):
                
                if(verbose):
                    print("Implementing social distancing at time t=" + str(t))
                
                #Social distancing is implemented
                G = socialDistance(G, alpha)
                distancingFlag = True #Don't implement social distancing again 
        
        zu = copy.deepcopy(zt) # nodes states for the previous time step (synchronous updates)

        #Loop over each node in the network
        for individual in G.nodes():
            #Choose whether a node interacts with its neighbors or a random node in the population
            #This section of code simulates S -> A, Y, E transitions
            
            #Choose random nodes in the network to interact with
            num_interactions_high = G.degree[individual] + 1
            num_interactions = np.random.randint(0, num_interactions_high)
            
            individual_status = zu[individual] #The status of the individual being evaluated 
            if np.random.rand(1)[0] < global_p:
                
                strangers = np.random.choice(G.nodes(), num_interactions)
                
                for stranger in strangers:
                    stranger_status = zu[stranger]
                    
                    #Check if the stranger is susceptible to infection
                    if(individual_status in infected_groups and stranger_status == 'S'):
                        #Pick a severity group for a potential infection
                        severity_group = np.random.choice(infected_groups, p=severity_pdf)

                        #Do a check on whether the stranger gets infected
                        if(np.random.rand(1)[0] < group_coefficients[severity_group][0]):
                            #Infect the stranger. Choose one of the groups to put them into
                            zt[stranger] = severity_group

                    #Check if the individual is susceptible to infection
                    if(stranger_status in infected_groups and individual_status == 'S'):
                        #Pick a severity group for a potential infection
                        severity_group = np.random.choice(infected_groups, p=severity_pdf)

                        if(np.random.rand(1)[0] < group_coefficients[severity_group][0]):
                            zt[individual] = severity_group
                        
            else:
                #TODO: Add functionality for proximity interactions here
                #Simulate the individual interacting with nodes that are adjacent to them
                neighbors = np.random.choice(list(G.neighbors(individual)), num_interactions)
                
                for neighbor in neighbors:
                    stranger = neighbor
                    stranger_status = zu[stranger]
                    #TODO: make this a nested function to keep it DRY and KISS
                    #Check if the stranger is susceptible to infection
                    if(individual_status in infected_groups and stranger_status == 'S'):
                        #Pick a severity group for a potential infection
                        severity_group = np.random.choice(infected_groups, p=severity_pdf)

                        #Do a check on whether the stranger gets infected
                        if(np.random.rand(1)[0] < group_coefficients[severity_group][0]):
                            #Infect the stranger. Choose one of the groups to put them into
                            zt[stranger] = severity_group

                    #Check if the individual is susceptible to infection
                    if(stranger_status in infected_groups and individual_status == 'S'):
                        #Pick a severity group for a potential infection
                        severity_group = np.random.choice(infected_groups, p=severity_pdf)

                        if(np.random.rand(1)[0] < group_coefficients[severity_group][0]):
                            zt[individual] = severity_group
            
            #Simulate recovery transitions. A,Y,E -> R
            if(zu[individual] in infected_groups):
                if(np.random.rand(1)[0] < group_coefficients[individual_status][1]):
                    zt[individual] = 'R'

        t += 1 # update clock
        
        #Optionally draw at each step
        if(draw):
            print(f'time step {t}')
            drawGz(G,zt)

        #Update the time series for Z
        #TODO: Update this for the different categories
        s_curr = 0
        a_curr = 0
        y_curr = 0
        e_curr = 0
        r_curr = 0
        for key,value in zt.items():
            if(value == 'S'):
                s_curr += 1
            elif(value == 'A'):
                a_curr += 1
            elif(value == 'Y'):
                y_curr += 1
            elif(value == 'E'):
                e_curr += 1
            elif(value == 'R'):
                r_curr += 1

        St.append(s_curr)
        At.append(a_curr)
        Yt.append(y_curr)
        Et.append(e_curr)
        Rt.append(r_curr)
        It.append(a_curr + y_curr + e_curr)
    
    return t, It, St, At, Yt, Et, Rt


#Plot results of the SAYER model simulation
def plotSAYER(t, It, St, At, Yt, Et, Rt, title='SAYER Model Totals'):
    x_time = np.arange(1, t+1)
    plot_line_width = 4
    colors = {'S':'#99FF99', 'A':'#FFB570', 'Y':'#FF99CC', 'E':'#FF3333', 'R':'#6666FF'}
    plt.plot(x_time, It, color="black", label="Total Infections", linewidth=plot_line_width)
    plt.plot(x_time, St, color=colors['S'], label="Susceptible", linewidth=plot_line_width)
    plt.plot(x_time, At, color=colors['A'], label="Asymptomatic", linewidth=plot_line_width)
    plt.plot(x_time, Yt, color=colors['Y'], label="Symptomatic", linewidth=plot_line_width)
    plt.plot(x_time, Et, color=colors['E'], label="Severe", linewidth=plot_line_width)
    plt.plot(x_time, Rt, color=colors['R'], label="Recovered/Removed", linewidth=plot_line_width)
    plt.grid(b=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Days")
    plt.ylabel("Number of Individuals")
    plt.title(title)
    plt.show()


# In[7]:


#Try out some beta and gamma values 
asymptomatic_beta = .8
asymptomatic_gamma = .2
symptomatic_beta = .5 
symptomatic_gamma = .3
severe_beta = .4
severe_gamma = .3

#Define the group_coefficients dictionary for the simulation
#This was chosen to avoid passing an additional 6 arguments to the sim function
group_coefficients = {'A':(asymptomatic_beta, asymptomatic_gamma), 'Y':(symptomatic_beta, symptomatic_gamma), 'E':(severe_beta, severe_gamma)}

#Other parameters for the simulation
initial_infected = 1
severity_pdf = [0.5, 0.42, 0.08]
global_p = 0.5

#Initialize G with an ER random graph and generate a healty population
def erGraph(n, c):
    p = c / (n-1)
    G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
    return G





# Now that we have a simulation that is capable of examining variables that we are interested in, we can now focus on generating graphs that are larger in order to examine the dynamics with more nodes. 
# 
# We will use the approach of the `nx.relaxed_caveman_graph` function to generate cliques that are connected together. However, the issue with `nx.relaxed_caveman_graph` is that each clique is fully connected, and then edges are randomly rewired with probability $p$. Instead, we wish to make each clique representative of the empirical proximity data, generate several cliques, and then randomly rewire each edge with probability $p$. Therefore, we will first concern ourselves with finding a model that adequately explains the empirical distribution for key statistics of the proximity data.
# 

#Generate a county graph using the chung-lu model for cliques
#Parameters:
#degree_sequence: The expected degree sequence for the chung_lu model
#l: The number of cliques in the network
#p: The probability of rewiring each edge in the network
def county_graph(degree_sequence, l, p, verbose=True):
    #Generate the first clique
    county_graph = nx.expected_degree_graph(degree_sequence, selfloops=False)
    
    #Add l-1 additional cliques to the existing graph
    if(verbose):
        print("Generating cliques")
    for _ in range(0, l-1):
        #Generate a clique to add to the existing county
        addition = nx.expected_degree_graph(degree_sequence, selfloops=False)
        
        #Force a disjoint union in the network
        county_graph = nx.disjoint_union(county_graph, addition)
        
    nodes = list(county_graph)
    
    #Loop over all edges and randomly rewire them with probability p
    if(verbose):
        print("Rewiring Edges")
    for (u,v) in county_graph.edges():
        #Choose whether to rewire with probability p
        if(np.random.rand(1)[0] < p):
            new_end = np.random.choice(nodes)
            
            if(county_graph.has_edge(u, new_end)):
                continue
                
            county_graph.remove_edge(u, v)
            county_graph.add_edge(u, new_end)
            
    if(verbose):
        print("Removing Disconnected Components")
    for component in list(nx.connected_components(county_graph)):
        if(len(component)<5):
            for node in component:
                county_graph.remove_node(node)
            
    return county_graph


def runSocialDistanceExperiment(G, global_p_param, infected_percent_arr, alpha_arr, initial_infected, severity_pdf, group_coefficients, plot=False):
    alphas_x = []
    percents_y = []
    peaks_infected = []
    peaks_asymptomatic = []
    peaks_symptomatic = []
    peaks_severe = []
    
    total_simulations = len(infected_percent_arr) * len(alpha_arr)

    count = 1
    
    for curr_percent in infected_percent_arr:
        for curr_alpha in alpha_arr:
            #Run the sayer simulation
            print("Running simulation number " + str(count) + " of " + str(total_simulations))
            print("Infected percent before social distancing: " + str(curr_percent))
            print("Probability of deleting edges with social distancing: " + str(curr_alpha))
            t, It, St, At, Yt, Et, Rt = runSAYER(G, initial_infected, severity_pdf, group_coefficients, global_p_param, curr_percent, curr_alpha, verbose=True, draw=False)
            print("========================================================================")
            alphas_x.append(curr_alpha)
            percents_y.append(curr_percent)
            peaks_infected.append(max(It))
            peaks_asymptomatic.append(max(At))
            peaks_symptomatic.append(max(Yt))
            peaks_severe.append(max(Et))
            
            count += 1
            
    return alphas_x, percents_y, peaks_infected, peaks_asymptomatic, peaks_symptomatic, peaks_severe

#Try out some beta and gamma values 
asymptomatic_beta = .8
asymptomatic_gamma = .2
symptomatic_beta = .5 
symptomatic_gamma = .3
severe_beta = .4
severe_gamma = .3

#Define the group_coefficients dictionary for the simulation
#This was chosen to avoid passing an additional 6 arguments to the sim function
group_coefficients = {'A':(asymptomatic_beta, asymptomatic_gamma), 'Y':(symptomatic_beta, symptomatic_gamma), 'E':(severe_beta, severe_gamma)}

#Other parameters for the simulation
initial_infected = 1
severity_pdf = [0.5, 0.42, 0.08]
global_p = 0.5

#Make a county graph
degree_sequence = [d for n, d in interaction_network.degree]
l = 20
q = 0.05
print("Graph generating with " + str(l) + " cliques and q=" + str(q))
county = county_graph(degree_sequence, l, q)
print("=============================================================")

nsamples = 25

infected_percent_arr = np.linspace(0, 0.5, nsamples)
alpha_arr = np.linspace(0, 1, nsamples)

nreps = 10

for i in range(nreps):
    print("Running repetition " + str(i+1) + " of " + str(nreps))
    alphas_x, percents_y, peaks_infected, peaks_asymptomatic, peaks_symptomatic, peaks_severe = runSocialDistanceExperiment(county, 
                                    global_p, 
                                    infected_percent_arr, 
                                    alpha_arr, initial_infected, 
                                    severity_pdf, 
                                    group_coefficients, 
                                    plot=False)

    #Write the data out to a file as a set of points
    timestr = time.strftime("%Y%m%d-%H%M%S") #Add the time to make each file unique
    infectionFile = open('data/maxinfected/' + timestr + '-rep' + str(i) + '.txt', 'w+')
    asymptomaticFile = open('data/maxasymptomatic/' + timestr + '-rep' + str(i) + '.txt', 'w+')
    symptomaticFile = open('data/maxsymptomatic/' + timestr + '-rep' + str(i) + '.txt', 'w+')
    severeFile = open('data/maxsevere/' + timestr + '-rep' + str(i) + '.txt', 'w+')

    for i in range(len(alphas_x)):
        #Prepare lines to be written to each file
        infectionline = str(alphas_x[i]) + " " + str(percents_y[i]) + " " + str(peaks_infected[i]) + "\n"
        asymptomaticline = str(alphas_x[i]) + " " + str(percents_y[i]) + " " + str(peaks_asymptomatic[i]) + "\n"
        symptomaticline = str(alphas_x[i]) + " " + str(percents_y[i]) + " " + str(peaks_symptomatic[i]) + "\n"
        severeline = str(alphas_x[i]) + " " + str(percents_y[i]) + " " + str(peaks_severe[i]) + "\n"

        infectionFile.write(infectionline)
        asymptomaticFile.write(asymptomaticline)
        symptomaticFile.write(symptomaticline)
        severeFile.write(severeline)
        

    infectionFile.close()
    asymptomaticFile.close()
    symptomaticFile.close()
    severeFile.close()

    print("**********************************************************")
