using Pkg
using Distributions
using Random
using Statistics
using ActiveInference
using Distributed
using Optim

addprocs(10)

@everywhere begin
    include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")
    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "stochastic")

    env = PrisonersDilemmaEnv()

    # Initialize all agents
    AlgoAgent_constructors = [
        TitForTatAgent,
        TitFor2TatsAgent,
        TwoTitsFor1TatAgent,
        NastyForgivingTFTAgent,
        PavlovianAgent,
        GrofmanAgent,
        GrimTriggerAgent,
        ()
    ]

    AlgoAgent_names = [
        "TitForTatAgent",
        "TitFor2TatsAgent",
        "TwoTitsFor1TatAgent",
        "NastyForgivingTFTAgent",
        "PavlovianAgent",
        "GrofmanAgent",
        "GrimTriggerAgent"
    ]

    function run_simulation(AlgoAgent_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
        # Initialize agents
        AIF_agent = init_AIF_agent_full(alpha, beta, lr_pB, fr_pB)
        AlgoAgent = AlgoAgent_constructor()

        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 10

        # Starting Observation
        obs1 = [1]
        obs2 = [1]

        actions_AIF_agent_store = []
        actions_AlgoAgent_store = []

        obs_AIF_agent_store = []
        obs_AlgoAgent_store = []

        AIF_EFE_store = []

        score_AIF_agent = 0
        score_AlgoAgent = 0

        # Run simulation
        for t in 1:N_TRIALS

            infer_states!(AIF_agent, obs1)
            update_AlgoAgent(AlgoAgent, obs2)

            if get_states(AIF_agent)["action"] !== missing
                QS_prev = get_history(AIF_agent)["posterior_states"][end-1]
                update_B!(AIF_agent, QS_prev)
            end

            infer_policies!(AIF_agent)
            
            action_AIF_agent = sample_action!(AIF_agent)
            action_AlgoAgent = choose_action_AlgoAgent(AlgoAgent)
            
            action_AIF_agent = Int(action_AIF_agent[1])
            push!(actions_AIF_agent_store, action_AIF_agent)
        
            action_AlgoAgent = Int(action_AlgoAgent[1])
            push!(actions_AlgoAgent_store, action_AlgoAgent)
        
            obs1, obs2, score_AIF_agent, score_AlgoAgent = trial(env, action_AIF_agent, action_AlgoAgent)
            obs1 = [findfirst(isequal(obs1), conditions)]
            obs2 = [findfirst(isequal(obs2), conditions)]
        
            push!(obs_AIF_agent_store, obs1)
            push!(obs_AlgoAgent_store, obs2)
        
        end
        
        return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, alpha = alpha, beta = beta, lr_pB = lr_pB, fr_pB = fr_pB)
    end
end

@model function model(AlgoAgent_constructors, AlgoAgent_names)
    # Priors for the parameters
    alpha ~ Normal(0.79, 0.2)
    beta ~ Normal(0.31, 0.1)
    lr_pB ~ Normal(0.81, 0.1)
    fr_pB ~ Normal(0.11, 0.05)

    # Initialize list to store scores
    scores_AIF_agent = []

    # Iterate over all AlgoAgents
    for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
        result = run_simulation(AlgoAgent_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
        score_AIF_agent = result[:score_AIF_agent]
        push!(scores_AIF_agent, score_AIF_agent)
    end

    # Likelihood of the observed data
    for score in scores_AIF_agent
        score ~ Normal(score, 1)  # Assuming normal likelihood with some noise
    end
end


# Collect observed scores
observed_scores = [score[:score_AIF_agent] for score in results]  # Ensure this matches your observed data format

# Define the Turing model
turing_model = model(AlgoAgent_constructors, AlgoAgent_names)

# Perform MCMC sampling
chain = sample(turing_model, NUTS(), 1000)

# Analyze Results
using StatsPlots
plot(chain)



