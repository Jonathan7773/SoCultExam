using ActiveInference
using Distributed
using DataFrames
using CSV
using Plots

addprocs(10)

@everywhere begin
    include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")
    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "deterministic",
                    "policy_len" => 2)

    env = PrisonersDilemmaEnv()

    function init_AIF_Agent()
        parameters_AIF = Dict{String, Real}(
        "lr_pB" => 0.59,
        "alpha" => 16
        )

        C = array_of_any_zeros(4)
        C[1][1] = 3.0 # CC
        C[1][2] = 1.0 # CD
        C[1][3] = 4.0 # DC
        C[1][4] = 2.0 # DD

        # C[1][1] = 2.0 # CC
        # C[1][2] = 4.0 # CD
        # C[1][3] = 1.0 # DC
        # C[1][4] = 3.0 # DD

        E = [2.0, 1.0]
        
        β = 1.28
        C[1] = softmax(C[1] * β)

        AIF_agent = init_aif(A_matrix, B_matrix;
                        C=C,
                        pB=pB,
                        settings=settings,
                        parameters=parameters_AIF,
                        verbose = false)
        return AIF_agent
    end

    # Initialize all agents
    agent_constructors = [
        TitForTatAgent,
        TitFor2TatsAgent,
        TwoTitsFor1TatAgent,
        NastyForgivingTFTAgent,
        PavlovianAgent,
        GrofmanAgent,
        GrimTriggerAgent,
        init_AIF_Agent
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
        agent1 = agent1_constructor()
        agent2 = agent2_constructor()

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

#CSV.write("ranked_scores.csv", ranked_scores_df)

# Winner mapping

function get_winner(agent1_name, agent2_name, score_agent1, score_agent2)
    if score_agent1 > score_agent2
        return agent1_name
    elseif score_agent2 > score_agent1
        return agent2_name
    else
        return "Draw"
    end
end

# Add a new column with the winner
results_df[!, :winner] = [get_winner(row.agent1_name, row.agent2_name, row.score_agent1, row.score_agent2) for row in eachrow(results_df)]

agents = unique(vcat(results_df.agent1_name, results_df.agent2_name))

# Initialize a dictionary to store win counts
win_counts = Dict(agent => 0 for agent in agents)

# Count the wins for each agent
for row in eachrow(results_df)
    winner = row.winner
    if winner != "Draw"
        win_counts[winner] += 1
    end
end

win_counts

agents_array = collect(keys(win_counts))
wins_array = collect(values(win_counts))

# Convert the win counts to a DataFrame for plotting
win_counts_df = DataFrame(agent = agents_array, wins = wins_array)

theme(:kk)
# Create the bar plot
WinPlot = bar(win_counts_df.agent, win_counts_df.wins, xlabel="Agent", ylabel="Number of Wins", title="Number of Wins per Agent", legend=false, ylim = (0, 7), xrotation=20, size = (800,400), color = :midnightblue, bg_color = :lightgoldenrod1, margin=20Plots.mm)


# Loser mapping

function get_loser(agent1_name, agent2_name, score_agent1, score_agent2)
    if score_agent1 < score_agent2
        return agent1_name
    elseif score_agent2 < score_agent1
        return agent2_name
    else
        return "Draw"
    end
end

# Add a new column with the winner
results_df[!, :loser] = [get_loser(row.agent1_name, row.agent2_name, row.score_agent1, row.score_agent2) for row in eachrow(results_df)]

agents = unique(vcat(results_df.agent1_name, results_df.agent2_name))

# Initialize a dictionary to store win counts
loser_counts = Dict(agent => 0 for agent in agents)

# Count the wins for each agent
for row in eachrow(results_df)
    loser = row.loser
    if loser != "Draw"
        loser_counts[loser] += 1
    end
end

loser_counts

agents_array = collect(keys(loser_counts))
loser_array = collect(values(loser_counts))

# Convert the win counts to a DataFrame for plotting
loser_counts_df = DataFrame(agent = agents_array, losses = loser_array)

theme(:kk)
# Create the bar plot
LossPlot = bar(loser_counts_df.agent, loser_counts_df.losses, xlabel="Agent", ylabel="Number of Losses", title="Number of Losses per Agent", legend=false, ylim = (0, 9), xrotation=20, size = (800,400), color = :midnightblue, bg_color = :lightgoldenrod1, margin=20Plots.mm)

##### Draws mapping #####
results_df

draw_counts = Dict(agent => 0 for agent in agents)

for row in eachrow(results_df)
    if row.winner == "Draw"
        draw_counts[row.agent1_name] += 1
        draw_counts[row.agent2_name] += 1
    end
end

agents_array = collect(keys(draw_counts))
draw_array = collect(values(draw_counts))

draw_counts_df = DataFrame(agent = agents_array, draws = draw_array)

DrawPlot = bar(draw_counts_df.agent, draw_counts_df.draws, xlabel="Agent", ylabel="Number of Draws", title="Number of Draws per Agent", legend=false, ylim = (0, 15), xrotation=20, size = (800,400), color = :midnightblue, bg_color = :lightgoldenrod1, margin=20Plots.mm)



WinPlot
LossPlot 
DrawPlot

CombinedWinsPlots = plot(WinPlot, LossPlot, DrawPlot, layout=(1,3), size = (1400, 600), margin=10Plots.mm)

#savefig(CombinedWinsPlots, "CombinedWinsPlots.svg")
