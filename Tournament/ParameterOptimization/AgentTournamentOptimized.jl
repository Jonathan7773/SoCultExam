using ActiveInference
using Distributed
using DataFrames

#addprocs(10)

@everywhere begin
    include(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\SoCultRepo\SoCultExam\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\SoCultRepo\SoCultExam\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\SoCultRepo\SoCultExam\EnvsAndAgents\GenerativeModel.jl")
    include(raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\SoCultRepo\SoCultExam\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "deterministic",
                    "policy_len" => 2)

    env = PrisonersDilemmaEnv()

    agent_params = Dict(
    "TitForTatAgent" => (0.4, 0.22, 1.0),
    "TitFor2TatsAgent" => (0.08, 0.73, 1.0),
    "TwoTitsFor1TatAgent" => (0.008, 1.0, 1.0),
    "NastyForgivingTFTAgent" => (0.62, 0.21, 1.0),
    "PavlovianAgent" => (0.04, 0.01, 1.0),
    "GrofmanAgent" => (1.96, 0.91, 1.0),
    "GrimTriggerAgent" => (0.04, 0.19, 1.0)
    )


    # Initialize all agents
    agent_constructors = [
        TitForTatAgent,
        TitFor2TatsAgent,
        TwoTitsFor1TatAgent,
        NastyForgivingTFTAgent,
        PavlovianAgent,
        GrofmanAgent,
        GrimTriggerAgent,
        init_AIF_agent_full_eksA
    ]

    agent_names = [
        "TitForTatAgent",
        "TitFor2TatsAgent",
        "TwoTitsFor1TatAgent",
        "NastyForgivingTFTAgent",
        "PavlovianAgent",
        "GrofmanAgent",
        "GrimTriggerAgent",
        "AIF_Agent"
    ]


    function run_simulation(agent1_constructor, agent2_constructor, agent1_name, agent2_name)
        # Initialize agents
        if agent1_name == "AIF_Agent"
            agent1 = agent1_constructor(agent_params[agent2_name]...)
        else
            agent1 = agent1_constructor()
        end

        if agent2_name == "AIF_Agent"
            agent2 = agent2_constructor(agent_params[agent1_name]...)
        else
            agent2 = agent2_constructor()
        end

        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 1000

        # Starting Observation
        obs1 = [1]
        obs2 = [1]

        actions_agent1_store = []
        actions_agent2_store = []

        obs_agent1_store = []
        obs_agent2_store = []

        score_agent1 = 0
        score_agent2 = 0

        # Run simulation
        for t in 1:N_TRIALS

            if agent1_name == "AIF_Agent"
                infer_states!(agent1, obs1)
            else
                update_AlgoAgent(agent1, obs1)
            end
        
            if agent2_name == "AIF_Agent"
                infer_states!(agent2, obs2)
            else
                update_AlgoAgent(agent2, obs2)
            end
        
            if agent1_name == "AIF_Agent"
                if get_states(agent1)["action"] !== missing
                    QS_prev = get_history(agent1)["posterior_states"][end-1]
                    update_B!(agent1, QS_prev)
                end
            end

            if agent2_name == "AIF_Agent"
                if get_states(agent2)["action"] !== missing
                    QS_prev = get_history(agent2)["posterior_states"][end-1]
                    update_B!(agent2, QS_prev)
                end
            end

            if agent1_name == "AIF_Agent"
                infer_policies!(agent1)
            end

            if agent2_name == "AIF_Agent"
                infer_policies!(agent2)
            end
            
            if agent1_name == "AIF_Agent"
                action_agent1 = sample_action!(agent1)
            else
                action_agent1 = choose_action_AlgoAgent(agent1)
            end
            action_agent1 = Int(action_agent1[1])
            push!(actions_agent1_store, action_agent1)
        
            if agent2_name == "AIF_Agent"
                action_agent2 = sample_action!(agent2)
            else
                action_agent2 = choose_action_AlgoAgent(agent2)
            end
            action_agent2 = Int(action_agent2[1])
            push!(actions_agent2_store, action_agent2)
        
            obs1, obs2, score_agent1, score_agent2 = trial(env, action_agent1, action_agent2)
            obs1 = [findfirst(isequal(obs1), conditions)]
            obs2 = [findfirst(isequal(obs2), conditions)]
        
            push!(obs_agent1_store, obs1)
            push!(obs_agent2_store, obs2)
        
        end
        
        return (agent1_name=agent1_name, agent2_name=agent2_name, score_agent1=score_agent1, score_agent2=score_agent2)
    end
end

results = []

# Create list of agent pairs
agent_pairs = [(agent1_constructor, agent2_constructor, agent1_name, agent2_name) for (agent1_constructor, agent1_name) in zip(agent_constructors, agent_names) for (agent2_constructor, agent2_name) in zip(agent_constructors, agent_names) if agent1_name != agent2_name]

@time begin
    results = @distributed (append!) for (agent1_constructor, agent2_constructor, agent1_name, agent2_name) in agent_pairs
        println("Running simulation: $agent1_name vs $agent2_name")
        result = run_simulation(agent1_constructor, agent2_constructor, agent1_name, agent2_name)
        [result]
    end
end

results_df = DataFrame(results)

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

agent_params = Dict(
    "TitForTatAgent" => (1.475, 0.036, 1.0),
    "TitFor2TatsAgent" => (1.495, 0.039, 1.0),
    "TwoTitsFor1TatAgent" => (0.005, 0.001, 1.0),
    "NastyForgivingTFTAgent" => (1.505, 0.037, 1.0),
    "PavlovianAgent" => (0.005, 0.001, 1.0),
    "GrofmanAgent" => (0.005, 0.001, 1.0),
    "GrimTriggerAgent" => (0.005, 0.001, 1.0)
)

using JLD2

df = DataFrame(results)

df_filtered = filter(row -> row.AlgoAgent_name == "GrimTriggerAgent", df)

ranked = sort!(df_filtered, :score_AIF_agent, rev = true)

