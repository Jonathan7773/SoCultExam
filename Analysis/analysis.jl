using Pkg
Pkg.activate("C:\\Users\\Samuel\\Desktop\\Julia_projects\\Julia-Development")
using Revise
using DataFrames, Serialization, Plots, LinearAlgebra
Pkg.develop(path="C:\\Users\\Samuel\\dev\\ActiveInference")
using ActiveInference

Pkg.add("JLD2")
using JLD2

data_path = "D:\\Stahovanie\\results.jld2"
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





















