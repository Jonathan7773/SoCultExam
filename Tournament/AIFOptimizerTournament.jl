using ActiveInference
using Distributed
using DataFrames
using Serialization

addprocs(10)

@everywhere begin
    include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")
    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "stochastic")

    parameters = Dict{String, Real}("lr_pB" => 0.1)

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

    function run_simulation(agent2_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
        # Initialize agents
        AIF_agent = init_AIF_agent_full(alpha, beta, lr_pB, fr_pB)
        AlgoAgent = agent2_constructor()

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

alphas = 1.0:2.0:32
betas = 0.5:0.5:5.5
lr_pBs = 0.1:0.1:1.0
fr_pBs = 0.1:0.1:1.0

length(alphas)*length(betas)*length(lr_pBs)*length(fr_pBs)*length(AlgoAgent_names)

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
    results = @distributed (append!) for alpha in alphas
        local_results = []
        for beta in betas
            for lr_pB in lr_pBs
                for fr_pB in fr_pBs
                    for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
                        #println("Running simulation: AIF_agent vs $AlgoAgent_name with alpha=$alpha, beta=$beta, lr_pB=$lr_pB, fr_pB=$fr_pB")
                        result = run_simulation(AlgoAgent_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
                        push!(local_results, result)
                    end
                end
            end
        end
        local_results
    end
end

results_df = DataFrame(results)

jls_filename = raw"SoCultRepo\SoCultExam\ResultData\results_df.jls"

open(jls_filename, "w") do io
    serialize(io, results_df)
end



SoCultRepo\SoCultExam\ResultData

agent_scores = Dict{String, Int64}()

for row in eachrow(results_df)
    agent1_name = row[:agent1_name]
    agent2_name = row[:agent2_name]
    score_agent1 = row[:score_agent1]
    score_agent2 = row[:score_agent2]

    if !haskey(agent_scores, agent1_name)
        agent_scores[agent1_name] = 0
    end
    if !haskey(agent_scores, agent2_name)
        agent_scores[agent2_name] = 0
    end

    agent_scores[agent1_name] += score_agent1
    agent_scores[agent2_name] += score_agent2
end

agent_names = collect(keys(agent_scores))
total_scores = collect(values(agent_scores))

total_scores_df = DataFrame(agent_name=agent_names, total_score=total_scores)

ranked_scores_df = sort(total_scores_df, :total_score, rev=true)

ranked_scores_df