using ActiveInference
using Distributed
using DataFrames
using Serialization
using Statistics
using Plots

#addprocs(60)

@everywhere begin
    include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")
    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "deterministic")

    function init_AIF_Agent()
        parameters_AIF = Dict{String, Real}(
        "lr_pB" => 0.5,
        "alpha" => 16
        )

        C = array_of_any_zeros(4)
        C[1][1] = 4.0 # CC
        C[1][2] = 1.0 # CD
        C[1][3] = 3.0 # DC
        C[1][4] = 2.0 # DD

        β = 1.5
        C[1] = softmax(C[1] * β)

        AIF_agent = init_aif(A_matrix, B_matrix;
                        C=C,
                        pB=pB,
                        settings=settings,
                        parameters=parameters_AIF,
                        verbose = false)
        return AIF_agent
    end

    env = PrisonersDilemmaEnv()

    # Initialize all agents
    AlgoAgent_constructors = [
        TitForTatAgent,
        ()
    ]

    AlgoAgent_names = [
        "TitForTatAgent"
    ]

    function run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, lr_pB)
        # Initialize agents
        AIF_agent = init_AIF_Agent()
        AlgoAgent = AlgoAgent_constructor()

        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 100

        # Starting Observation
        obs1 = [1]
        obs2 = [1]

        actions_AIF_agent_store = []
        actions_AlgoAgent_store = []

        obs_AIF_agent_store = []
        obs_AlgoAgent_store = []

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
        
        return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, beta = beta, lr_pB = lr_pB)
    end
end

learning_rates = 0.1:0.1:1.0
betas = 0.1:0.1:5.5

# List of AlgoAgent constructors and names
AlgoAgent_constructors = [
    TitForTatAgent
]

AlgoAgent_names = [
    "TitForTatAgent"
]

results = []

@time begin
    results = @distributed (append!) for beta in betas
        local_results = []
        println("Running simulation: beta=$beta")
        for learning_rate in learning_rates
            for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
                result = run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, learning_rate)
                push!(local_results, result)
            end
        end
        local_results
    end
end

results_df = DataFrame(results)

results_df[!, :total_reward] = results_df[!, :score_AIF_agent] .+ results_df[!, :score_AlgoAgent]

x = unique(results_df[!, :lr_pB])
y = unique(results_df[!, :beta])

pivot_df = unstack(results_df, :beta, :lr_pB, :score_AIF_agent)
z = Matrix(pivot_df[:, Not(:beta)])

heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Accumulated Total Reward", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)

kk

############################ Test with all agents #######################################

@everywhere begin
    include("../EnvsAndAgents/AIFInitFunctions.jl")
    include("../EnvsAndAgents/AlgoAgentsFunctions.jl")
    include("../EnvsAndAgents/GenerativeModel.jl")
    include("../EnvsAndAgents/PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "deterministic")

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

    function run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, lr_pB)
        # Initialize agents
        AIF_agent = init_AIF_agent_beta_lr_pB(beta, lr_pB)
        AlgoAgent = AlgoAgent_constructor()

        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 100

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
        
        return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, beta = beta, lr_pB = lr_pB)
    end
end

learning_rates = 0.01:0.01:1.0
betas = 0.1:0.1:5.5

# List of AlgoAgent constructors and names
AlgoAgent_constructors = [
    TitForTatAgent,
    TitFor2TatsAgent,
    TwoTitsFor1TatAgent,
    NastyForgivingTFTAgent,
    PavlovianAgent,
    GrofmanAgent,
    GrimTriggerAgent
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

results = []

@time begin
    results = @distributed (append!) for beta in betas
        local_results = []
        println("Running simulation: beta=$beta")
        for learning_rate in learning_rates
            for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
                result = run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, learning_rate)
                push!(local_results, result)
            end
        end
        local_results
    end
end

results_df = DataFrame(results)

results_df[!, :total_reward] = results_df[!, :score_AIF_agent] .+ results_df[!, :score_AlgoAgent]

x = unique(results_df[!, :lr_pB])
y = unique(results_df[!, :beta])

pivot_df = unstack(results_df, :beta, :lr_pB, :score_AIF_agent; combine=sum)
z = Matrix(pivot_df[:, Not(:beta)])

heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Accumulated Total Reward", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)
