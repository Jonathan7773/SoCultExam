using Plots
using DataFrames


include("EFETrackingC.jl")
include("EFETrackingD.jl")


plot_data_C
plot_data_D

sort!(plot_data_C, [:lr_pB, :time_step])
sort!(plot_data_D, [:lr_pB, :time_step])

EFE_difference = plot_data_D.average_EFE .- plot_data_C.average_EFE

plot_data_diff = DataFrame(
    lr_pB = plot_data_C.lr_pB,
    time_step = plot_data_C.time_step,
    EFE_difference = EFE_difference
)

plot_data_diff

res = 0.0:10:1000

filtered_plot_diff = filter(row -> row.time_step in res, plot_data_diff)

plt = plot(title="Average difference of EFE (D-C) for Different lr_pB",
           xlabel="Time Step",
           ylabel="Average EFE difference")

for lr_pB in unique(filtered_plot_diff.lr_pB)
    lr_pB_data = filtered_plot_diff[filtered_plot_diff.lr_pB .== lr_pB, :]
    plot!(plt, lr_pB_data.time_step, lr_pB_data.EFE_difference, label="lr_pB = $lr_pB")
end
theme(:juno)

display(plt)

