using ActiveInference
using Distributed
using DataFrames
using Serialization
using Statistics
using Plots

include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
include(raw"..\EnvsAndAgents\GenerativeModel.jl")
include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

settings = Dict("use_param_info_gain" => false,
                "use_states_info_gain" => false,
                "action_selection" => "deterministic",
                "policy_len" => 2)


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

env = PrisonersDilemmaEnv()

AIF_agent = init_AIF_Agent()
AlgoAgent = TwoTitsFor1TatAgent()

N_TRIALS = 300

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
AIF_agent.B[1]
score_AIF_agent, score_AlgoAgent

action_matrix = [actions_AIF_agent_store'; actions_AlgoAgent_store']

cmap = [:lightgoldenrod1, :midnightblue]

# Plot the heatmap
TTF1TPlot = heatmap(action_matrix, color=cmap,
            clims = (1,2),
            legend=false, ylabel="",
            yticks=(1:2, ["AIF", "TwoTitsFor1Tat"]),
            xticks=(0:10:160),
            size=(1000, 100))

AIF_agent.action

CombinedAlgoPlot = plot(TFTPlot, NFTFTPlot, TTF2TPlot, PavPlot, GrofPlot, layout = (5, 1), size = (1000, 700))

annotate!(CombinedAlgoPlot, 0.5, 1.05, text("AIF agent vs Algorithmic Agents", :center, 12))

CombinedAlgoPlot = plot(CombinedAlgoPlot, title = "AIF agent vs Algorithmic Agents")



savefig(CombinedAlgoPlot, raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\SoCultRepo\SoCultExam\ResultData\Plots\CombinedAlgoPlot.svg")

labels = ["r_CC", "r_CD", "r_DC", "r_DD"]

theme(:lightgoldenrod1)

betaplot = bar(labels, AIF_agent.C[1], legend=false, ylabel="", xlabel="", title="β = 1.28", color = :midnightblue, ylim = 0:1, bg_color = :lightgoldenrod1)
betapointfiveplot = bar(labels, AIF_agent.C[1], legend=false, ylabel="", xlabel="", title="β = 0.5", color = :midnightblue, ylim = 0:1, bg_color = :lightgoldenrod1)
betaoneplot = bar(labels, AIF_agent.C[1], legend=false, ylabel="", xlabel="", title="β = 1.0", color = :midnightblue, ylim = 0:1, bg_color = :lightgoldenrod1)
betaonepointfiveplot = bar(labels, AIF_agent.C[1], legend=false, ylabel="", xlabel="", title="β = 1.5", color = :midnightblue, ylim = 0:1, bg_color = :lightgoldenrod1)

combinebetaplot = plot(betapointoneplot, betapointfiveplot, betaoneplot, betaonepointfiveplot, size = (1000, 700))
savefig(betaplot, raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\SocCult\SoCultRepo\SoCultExam\ResultData\Plots\betaplot.svg")