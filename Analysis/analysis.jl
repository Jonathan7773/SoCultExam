using Pkg
Pkg.activate("C:\\Users\\Samuel\\Desktop\\Julia_projects\\Julia-Development")
using Revise
using DataFrames, Serialization, Plots, LinearAlgebra
Pkg.develop(path="C:\\Users\\Samuel\\dev\\ActiveInference")
using ActiveInference
using Statistics
using StatsBase
using StatsPlots
using JLD2

data_path = "results.jld2"
data = JLD2.load(data_path)

raw_data = data["results"]

df = DataFrame(raw_data)

describe(df)
unique(df[!,"AlgoAgent_name"])

df_titfortat = filter(row -> row.AlgoAgent_name == "TitForTatAgent", df)
df_titfor2tats = filter(row -> row.AlgoAgent_name == "TitFor2TatsAgent", df)
df_twotitsfor1tat = filter(row -> row.AlgoAgent_name == "TwoTitsFor1TatAgent", df)
df_nastyforgivingtft = filter(row -> row.AlgoAgent_name == "NastyForgivingTFTAgent", df)
df_pavlovian = filter(row -> row.AlgoAgent_name == "PavlovianAgent", df)
df_grofman = filter(row -> row.AlgoAgent_name == "GrofmanAgent", df)
df_grimtrigger = filter(row -> row.AlgoAgent_name == "GrimTriggerAgent", df)

################################################### Rewards AIF agent ################################################### 

############ TIT FOR TAT HEATMAP ###########
pivot_df = unstack(df_titfortat, :beta, :lr_pB, :score_AIF_agent)
x = unique(df_titfortat[!, :lr_pB])
y = unique(df_titfortat[!, :beta])
z = Matrix(pivot_df[:, Not(:beta)])
heatmap(x, y, z, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. TitForTat", colorbar_title="Accumulated Reward",
         color=:inferno, size =(800,700))

############ TitFor2Tats HEATMAP ###########
pivot_df_titfor2tats = unstack(df_titfor2tats, :beta, :lr_pB, :score_AIF_agent)
x_titfor2tats = unique(df_titfor2tats[!, :lr_pB])
y_titfor2tats = unique(df_titfor2tats[!, :beta])
z_titfor2tats = Matrix(pivot_df_titfor2tats[:, Not(:beta)])
heatmap(x_titfor2tats, y_titfor2tats, z_titfor2tats, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. TitFor2TatsAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ TwoTitsFor1Tat HEATMAP ###########
pivot_df_twotitsfor1tat = unstack(df_twotitsfor1tat, :beta, :lr_pB, :score_AIF_agent)
x_twotitsfor1tat = unique(df_twotitsfor1tat[!, :lr_pB])
y_twotitsfor1tat = unique(df_twotitsfor1tat[!, :beta])
z_twotitsfor1tat = Matrix(pivot_df_twotitsfor1tat[:, Not(:beta)])
heatmap(x_twotitsfor1tat, y_twotitsfor1tat, z_twotitsfor1tat, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. TwoTitsFor1TatAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ NastyForgivingTFT HEATMAP ###########
pivot_df_nastyforgivingtft = unstack(df_nastyforgivingtft, :beta, :lr_pB, :score_AIF_agent)

x_nastyforgivingtft = unique(df_nastyforgivingtft[!, :lr_pB])
y_nastyforgivingtft = unique(df_nastyforgivingtft[!, :beta])
z_nastyforgivingtft = Matrix(pivot_df_nastyforgivingtft[:, Not(:beta)])
heatmap(x_nastyforgivingtft, y_nastyforgivingtft, z_nastyforgivingtft, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. NastyForgivingTFTAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))
############ Pavlovian HEATMAP ###########
pivot_df_pavlovian = unstack(df_pavlovian, :beta, :lr_pB, :score_AIF_agent)

x_pavlovian = unique(df_pavlovian[!, :lr_pB])
y_pavlovian = unique(df_pavlovian[!, :beta])
z_pavlovian = Matrix(pivot_df_pavlovian[:, Not(:beta)])
heatmap(x_pavlovian, y_pavlovian, z_pavlovian, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. PavlovianAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))
############ Grofman HEATMAP ###########
pivot_df_grofman = unstack(df_grofman, :beta, :lr_pB, :score_AIF_agent)
x_grofman = unique(df_grofman[!, :lr_pB])
y_grofman = unique(df_grofman[!, :beta])
z_grofman = Matrix(pivot_df_grofman[:, Not(:beta)])
heatmap(x_grofman, y_grofman, z_grofman, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. GrofmanAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))
############ GrimTrigger HEATMAP ###########
pivot_df_grimtrigger = unstack(df_grimtrigger, :beta, :lr_pB, :score_AIF_agent)

x_grimtrigger = unique(df_grimtrigger[!, :lr_pB])
y_grimtrigger = unique(df_grimtrigger[!, :beta])
z_grimtrigger = Matrix(pivot_df_grimtrigger[:, Not(:beta)])
heatmap(x_grimtrigger, y_grimtrigger, z_grimtrigger, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of AIF Agent vs. GrimTriggerAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

################################################### Rewards Algos ################################################### 

############ TIT FOR TAT HEATMAP ###########
pivot_df = unstack(df_titfortat, :beta, :lr_pB, :score_AlgoAgent)
x = unique(df_titfortat[!, :lr_pB])
y = unique(df_titfortat[!, :beta])
z = Matrix(pivot_df[:, Not(:beta)])
heatmap(x, y, z, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of TitForTat Agent", colorbar_title="Accumulated Reward",
        color=:inferno, size =(800,700))

############ TitFor2Tats HEATMAP ###########
pivot_df_titfor2tats = unstack(df_titfor2tats, :beta, :lr_pB, :score_AlgoAgent)
x_titfor2tats = unique(df_titfor2tats[!, :lr_pB])
y_titfor2tats = unique(df_titfor2tats[!, :beta])
z_titfor2tats = Matrix(pivot_df_titfor2tats[:, Not(:beta)])
heatmap(x_titfor2tats, y_titfor2tats, z_titfor2tats, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of TitFor2TatsAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ TwoTitsFor1Tat HEATMAP ###########
pivot_df_twotitsfor1tat = unstack(df_twotitsfor1tat, :beta, :lr_pB, :score_AlgoAgent)
x_twotitsfor1tat = unique(df_twotitsfor1tat[!, :lr_pB])
y_twotitsfor1tat = unique(df_twotitsfor1tat[!, :beta])
z_twotitsfor1tat = Matrix(pivot_df_twotitsfor1tat[:, Not(:beta)])
heatmap(x_twotitsfor1tat, y_twotitsfor1tat, z_twotitsfor1tat, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of TwoTitsFor1TatAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ NastyForgivingTFT HEATMAP ###########
pivot_df_nastyforgivingtft = unstack(df_nastyforgivingtft, :beta, :lr_pB, :score_AlgoAgent)
x_nastyforgivingtft = unique(df_nastyforgivingtft[!, :lr_pB])
y_nastyforgivingtft = unique(df_nastyforgivingtft[!, :beta])
z_nastyforgivingtft = Matrix(pivot_df_nastyforgivingtft[:, Not(:beta)])
heatmap(x_nastyforgivingtft, y_nastyforgivingtft, z_nastyforgivingtft, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of NastyForgivingTFTAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ Pavlovian HEATMAP ###########
pivot_df_pavlovian = unstack(df_pavlovian, :beta, :lr_pB, :score_AlgoAgent)
x_pavlovian = unique(df_pavlovian[!, :lr_pB])
y_pavlovian = unique(df_pavlovian[!, :beta])
z_pavlovian = Matrix(pivot_df_pavlovian[:, Not(:beta)])
heatmap(x_pavlovian, y_pavlovian, z_pavlovian, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of PavlovianAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ Grofman HEATMAP ###########
pivot_df_grofman = unstack(df_grofman, :beta, :lr_pB, :score_AlgoAgent)
x_grofman = unique(df_grofman[!, :lr_pB])
y_grofman = unique(df_grofman[!, :beta])
z_grofman = Matrix(pivot_df_grofman[:, Not(:beta)])
heatmap(x_grofman, y_grofman, z_grofman, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of GrofmanAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

############ GrimTrigger HEATMAP ###########
pivot_df_grimtrigger = unstack(df_grimtrigger, :beta, :lr_pB, :score_AlgoAgent)
x_grimtrigger = unique(df_grimtrigger[!, :lr_pB])
y_grimtrigger = unique(df_grimtrigger[!, :beta])
z_grimtrigger = Matrix(pivot_df_grimtrigger[:, Not(:beta)])
heatmap(x_grimtrigger, y_grimtrigger, z_grimtrigger, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Accumulated Reward of GrimTriggerAgent", colorbar_title="Accumulated Reward",
        color=:inferno, size=(800,700))

################################################### TOTAL ACCUMULATED SCORES ################################################### 

############ TIT FOR TAT HEATMAP ###########
df_titfortat.TotalAccumulatedReward = df_titfortat.score_AlgoAgent .+ df_titfortat.score_AIF_agent
pivot_df = unstack(df_titfortat, :beta, :lr_pB, :TotalAccumulatedReward)
x = unique(df_titfortat[!, :lr_pB])
y = unique(df_titfortat[!, :beta])
z = Matrix(pivot_df[:, Not(:beta)])
heatmap(x, y, z, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - TitForTat", colorbar_title="Total Accumulated Reward",
        color=:inferno, size =(800,700))

############ TitFor2Tats HEATMAP ###########
df_titfor2tats.TotalAccumulatedReward = df_titfor2tats.score_AlgoAgent .+ df_titfor2tats.score_AIF_agent
pivot_df_titfor2tats = unstack(df_titfor2tats, :beta, :lr_pB, :TotalAccumulatedReward)
x_titfor2tats = unique(df_titfor2tats[!, :lr_pB])
y_titfor2tats = unique(df_titfor2tats[!, :beta])
z_titfor2tats = Matrix(pivot_df_titfor2tats[:, Not(:beta)])
heatmap(x_titfor2tats, y_titfor2tats, z_titfor2tats, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - TitFor2TatsAgent", colorbar_title="Total Accumulated Reward",
        color=:inferno, size=(800,700))

############ TwoTitsFor1Tat HEATMAP ###########
df_twotitsfor1tat.TotalAccumulatedReward = df_twotitsfor1tat.score_AlgoAgent .+ df_twotitsfor1tat.score_AIF_agent
pivot_df_twotitsfor1tat = unstack(df_twotitsfor1tat, :beta, :lr_pB, :TotalAccumulatedReward)
x_twotitsfor1tat = unique(df_twotitsfor1tat[!, :lr_pB])
y_twotitsfor1tat = unique(df_twotitsfor1tat[!, :beta])
z_twotitsfor1tat = Matrix(pivot_df_twotitsfor1tat[:, Not(:beta)])
heatmap(x_twotitsfor1tat, y_twotitsfor1tat, z_twotitsfor1tat, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - TwoTitsFor1TatAgent", colorbar_title="Total Accumulated Reward",
        color=:inferno, size=(800,700))


############ NastyForgivingTFT HEATMAP ###########
df_nastyforgivingtft.TotalAccumulatedReward = df_nastyforgivingtft.score_AlgoAgent .+ df_nastyforgivingtft.score_AIF_agent
pivot_df_nastyforgivingtft = unstack(df_nastyforgivingtft, :beta, :lr_pB, :TotalAccumulatedReward)
x_nastyforgivingtft = unique(df_nastyforgivingtft[!, :lr_pB])
y_nastyforgivingtft = unique(df_nastyforgivingtft[!, :beta])
z_nastyforgivingtft = Matrix(pivot_df_nastyforgivingtft[:, Not(:beta)])
heatmap(x_nastyforgivingtft, y_nastyforgivingtft, z_nastyforgivingtft, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - NastyForgivingTFTAgent", colorbar_title="Total Accumulated Reward",
        color=:inferno, size=(800,700))

############ Pavlovian HEATMAP ###########
df_pavlovian.TotalAccumulatedReward = df_pavlovian.score_AlgoAgent .+ df_pavlovian.score_AIF_agent
pivot_df_pavlovian = unstack(df_pavlovian, :beta, :lr_pB, :TotalAccumulatedReward)
x_pavlovian = unique(df_pavlovian[!, :lr_pB])
y_pavlovian = unique(df_pavlovian[!, :beta])
z_pavlovian = Matrix(pivot_df_pavlovian[:, Not(:beta)])
heatmap(x_pavlovian, y_pavlovian, z_pavlovian, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - PavlovianAgent", colorbar_title="Total Accumulated Reward",
        color=:inferno, size=(800,700))

############ Grofman HEATMAP ###########
df_grofman.TotalAccumulatedReward = df_grofman.score_AlgoAgent .+ df_grofman.score_AIF_agent
pivot_df_grofman = unstack(df_grofman, :beta, :lr_pB, :TotalAccumulatedReward)
x_grofman = unique(df_grofman[!, :lr_pB])
y_grofman = unique(df_grofman[!, :beta])
z_grofman = Matrix(pivot_df_grofman[:, Not(:beta)])
heatmap(x_grofman, y_grofman, z_grofman, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - GrofmanAgent", colorbar_title="Total Accumulated Reward",
        color=:inferno, size=(800,700))

############ GrimTrigger HEATMAP ###########
df_grimtrigger.TotalAccumulatedReward = df_grimtrigger.score_AlgoAgent .+ df_grimtrigger.score_AIF_agent
pivot_df_grimtrigger = unstack(df_grimtrigger, :beta, :lr_pB, :TotalAccumulatedReward)
x_grimtrigger = unique(df_grimtrigger[!, :lr_pB])
y_grimtrigger = unique(df_grimtrigger[!, :beta])
z_grimtrigger = Matrix(pivot_df_grimtrigger[:, Not(:beta)])
heatmap(x_grimtrigger, y_grimtrigger, z_grimtrigger, xlabel="lr_pB (η)", ylabel="Beta (β)", 
        title="Total Accumulated Reward - GrimTriggerAgent", colorbar_title="Total Accumulated Reward",
        color=:inferno, size=(800,700))


################################################### More Analysis ################################################### 

function create_combined_long_form()
    df_long = DataFrame()
    
    agent_dfs = [
        (df_titfortat, "TitForTatAgent"),
        (df_titfor2tats, "TitFor2TatsAgent"),
        (df_twotitsfor1tat, "TwoTitsFor1TatAgent"),
        (df_nastyforgivingtft, "NastyForgivingTFTAgent"),
        (df_pavlovian, "PavlovianAgent"),
        (df_grofman, "GrofmanAgent"),
        (df_grimtrigger, "GrimTriggerAgent")
    ]
    
    for (agent_df, agent_name) in agent_dfs
        temp_df = DataFrame(
            AgentType = repeat([agent_name], inner=2*size(agent_df, 1)),
            Score = vcat(agent_df.score_AlgoAgent, agent_df.score_AIF_agent),
            Category = vcat(fill("AlgoAgent", size(agent_df, 1)), fill("AIF Agent", size(agent_df, 1)))
        )
        df_long = vcat(df_long, temp_df)
    end
    
    return df_long
end

df_combined_long = create_combined_long_form()

@df df_combined_long groupedboxplot(:AgentType, :Score, group=:Category, legend=:topleft,
                             title="Scores Comparison: Algorithms vs. AIF Agents",
                             ylabel="Score", rotation=22.5, color=[:teal :red], size = (700,600))


#################### BETA (β) Parameter on Reward ####################
learning_rates = 0.1:0.1:1.0
colors = [:darkgreen, :green, :green3, :green2, :springgreen3, :springgreen2, :mediumspringgreen, :springgreen1, :seagreen2, :aqua]

p = plot(xlabel="Beta (β)", ylabel="Score of AIF Agent",
         title="GrimTrigger", legend=:bottomright)

for (i, lr) in enumerate(learning_rates)
    subset_titfortat = filter(row -> row.lr_pB == lr, df_grimtrigger)
    plot!(p, subset_titfortat.beta, subset_titfortat.score_AIF_agent, label="lr_pB = $lr", seriestype=:line, linewidth=2,color=colors[i], size = (700,500) 
    )
end

display(p)

#################### Learning Rate (η) Parameter on Reward ####################

beta_values = 0.5:0.5:3.0
colors_betas = ["#2c105c", :midnightblue, "#a52c60", "#f17c4e", "#fdca26", :yellow2]

p = plot(xlabel="Learning Rate (lr_pB)", ylabel="Score of AIF Agent",
         title="TitForTat", legend=:bottomright, size=(700,500))

for (i, beta) in enumerate(beta_values)
    subset_grimtrigger = filter(row -> row.beta == beta, df_titfortat)
    plot!(p, subset_grimtrigger.lr_pB, subset_grimtrigger.score_AIF_agent, label="beta = $beta", seriestype=:line, linewidth=2, color=colors_betas[i])
end

display(p)


(df_grimtrigger, "GrimTriggerAgent")

using ActiveInference

C = array_of_any_zeros(4)
C[1][1] = 3.0 # CC
C[1][2] = 1.0 # CD
C[1][3] = 4.0 # DC
C[1][4] = 2.0 # DD

β = 0.5
C[1] = softmax(C[1] * β)

bar(C[1], xlabel="Preferences", ylabel="", title="Softmax-Scaled Preferences β =.5", 
    xticks=(1:4, ["CC", "CD", "DC", "DD"]), tickfontsize=12,label=false,yticks=0:0.1:1.0, color=:darkorange2, linecolor=:darkorange2)


bar(C[1], xlabel="Preferences", ylabel="", title="Softmax-Scaled Preferences β = 0.5", 
xticks=(1:4, ["CC", "CD", "DC", "DD"]), tickfontsize=12,label=false,yticks=0:0.1:1.0,ylim=(0.0,1.0), color="#2c105c", linecolor="#2c105c")
