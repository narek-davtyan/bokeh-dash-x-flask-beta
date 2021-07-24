import pandas as pd
import numpy as np

# Import multiprocessing libraries
from pandarallel import pandarallel

# Initialization
pandarallel.initialize()

# Analytics
# import timeit

# Load data locally
df_orig = pd.read_excel(r'result_data_x.xlsx', names=['index', 'type', 'date', 'code', \
    'filter_one', 'filter_two', 'filter_three', 'filter_four', 'recommendation', 'easiness', 'overall', 'question_one', \
    'rec_sc', 'eas_sc', 'sentiment', 'lang', 'question_one_filtered_lemmas'])
# Load data on server
# df_orig = pd.read_excel(r'bokeh-dash-x/result_data_x.xlsx', names=['index', 'type', 'date', 'code', \
#     'filter_one', 'filter_two', 'filter_three', 'filter_four', 'recommendation', 'easiness', 'overall', 'question_one', \
#     'rec_sc', 'eas_sc', 'sentiment', 'lang', 'question_one_filtered_lemmas'])

# Transform filtered lemmas string into list of strings
df_orig['question_one_filtered_lemmas'] = df_orig.loc[~df_orig['question_one_filtered_lemmas'].isna()]['question_one_filtered_lemmas'].apply(lambda x: x[2:-2].split("', '"))

# Create dictionary of all plots, filter lock, filters
general_dict = {}

# Set initial filters to all
general_dict['full_name_filter_list'] = sorted(df_orig.filter_one.unique().tolist())
general_dict['full_service_filter_list'] = sorted(df_orig.filter_two.unique().tolist())
general_dict['full_factory_filter_list'] = sorted(df_orig.filter_three.unique().tolist())
general_dict['full_segment_filter_list'] = sorted(df_orig.filter_four.unique().tolist())
general_dict['type_list'] = sorted(df_orig.type.unique())
general_dict['individual_filters'] = {}
general_dict['individual_df'] = {}
general_dict['individual_filtered_df'] = {}
general_dict['individual_filtered_df_bary'] = {}
general_dict['bary_points'] = {}
general_dict['bary_p'] = {}
general_dict['individual_filtered_df_bary_p'] = {}
general_dict['individual_service_filter_list'] = {}

general_dict['typed_name_filter_list'] = {}
general_dict['typed_service_filter_list'] = {}
general_dict['typed_factory_filter_list'] = {}
general_dict['typed_segment_filter_list'] = {}
for type_filter in general_dict['type_list']:
    # general_dict['individual_filters'][type_filter] = np.concatenate((df_orig.filter_one.unique(),df_orig.filter_two.unique(),df_orig.filter_three.unique(),df_orig.filter_four.unique())).tolist() + [type_filter]
    general_dict['individual_filters'][type_filter] = np.concatenate((df_orig.filter_one.unique(),df_orig.filter_two.unique(),df_orig.filter_three.unique(),df_orig.filter_four.unique())).tolist() + [type_filter]
    general_dict['individual_df'][type_filter] = df_orig.loc[(df_orig['type'].isin([type_filter]))]
    general_dict['individual_filtered_df'][type_filter] = df_orig.loc[(df_orig['type'].isin([type_filter]))]
    general_dict['individual_service_filter_list'][type_filter] = sorted(general_dict['individual_df'][type_filter].filter_two.unique().tolist())
    
    general_dict['typed_name_filter_list'][type_filter] = general_dict['full_name_filter_list']
    general_dict['typed_service_filter_list'][type_filter] = general_dict['full_service_filter_list']
    general_dict['typed_factory_filter_list'][type_filter] = general_dict['full_factory_filter_list']
    general_dict['typed_segment_filter_list'][type_filter] = general_dict['full_segment_filter_list']

general_dict['freq_df'] = {}
general_dict['freq_words_slice'] = {}
general_dict['freq_source'] = {}
general_dict['d_pv'], general_dict['d_uv'], general_dict['d_nv'] = (dict(),dict(),dict())
general_dict['wordcloud'], general_dict['words_plot'] = (dict(),dict())
general_dict['dict_freq_pv'], general_dict['d_freq_pv'] = (dict(),dict())

###################################################################################
################################## Visual Imports #################################
from bokeh.models import ColumnDataSource, Callback, Toggle, BoxAnnotation, LabelSet, Label, HoverTool, DataTable, TableColumn, Image, TapTool, Tap, HBar, Plot, Div, CDSView, GroupFilter, MultiChoice, MultiSelect, CheckboxButtonGroup, BooleanFilter, IndexFilter, RadioButtonGroup, Button, CustomJS
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, Spacer, gridplot
###################################################################################
###################################################################################

###################################################################################
################################## Common Methods #################################

def update_filters():
    # common_filters = [general_dict['filter_3'].labels[i] for i in general_dict['filter_3'].active] + general_dict['filter_1'].value + general_dict['filter_4'].value
    common_filters = general_dict['filter_4'].value
    for type_filter in general_dict['type_list']:
        # uncommon_filters = general_dict['individual_filters_vis'][type_filter].value
        # general_dict['individual_filters'][type_filter] = common_filters + uncommon_filters# + [type_filter]
        general_dict['individual_filters'][type_filter] = common_filters

# def update_filter(type_filter):
#     common_filters = [general_dict['filter_3'].labels[i] for i in general_dict['filter_3'].active] + general_dict['filter_1'].value + general_dict['filter_4'].value
#     uncommon_filters = general_dict['individual_filters_vis'][type_filter].value
#     general_dict['individual_filters'][type_filter] = common_filters + uncommon_filters# + [type_filter]

def filter_df(type_filter):

    filter_list = general_dict['individual_filters'][type_filter]

    # general_dict['individual_filtered_df'][type_filter] = general_dict['individual_df'][type_filter].loc[(general_dict['individual_df'][type_filter]['filter_one'].isin(filter_list) & general_dict['individual_df'][type_filter]['filter_two'].isin(filter_list) & general_dict['individual_df'][type_filter]['filter_three'].isin(filter_list) & general_dict['individual_df'][type_filter]['filter_four'].isin(filter_list))].copy()
    general_dict['individual_filtered_df'][type_filter] = general_dict['individual_df'][type_filter].loc[(general_dict['individual_df'][type_filter]['filter_four'].isin(filter_list))].copy()

def filter_dfs():
    for type_filter in general_dict['type_list']:
        filter_df(type_filter)

###################################################################################
############################## Visual 1 - Points Plot #############################


#---------------------------------------------------------------------------------#
#------------------------------- Static Background -------------------------------#

def create_points_plot_layout(points_plot_name):
    # Create points plot figure
    general_dict[points_plot_name] = figure(x_range=(0, 10), y_range=(0, 10), plot_width=300, plot_height=300, sizing_mode='scale_width', match_aspect=True, tools=['tap'], output_backend="webgl", title=points_plot_name.split('_')[0], title_location="above")#, lod_factor=1)
    general_dict[points_plot_name].title.align = "center"
    general_dict[points_plot_name].title.text_font_size = "16px"
    general_dict[points_plot_name].title.text_font_style='bold'

    # Hide real grid
    general_dict[points_plot_name].xgrid.grid_line_color = None
    general_dict[points_plot_name].ygrid.grid_line_color = None

    # Define grid lines
    general_dict[points_plot_name].xaxis.ticker = list(range(11))
    general_dict[points_plot_name].yaxis.ticker = list(range(11))

    # Create color zones
    ba5 = BoxAnnotation(bottom=9, top=10, left=0, right=10, fill_alpha=0.3, fill_color='#538d22', level='underlay', line_color=None) # green
    ba4 = BoxAnnotation(bottom=0, top=9, left=9, right=10, fill_alpha=0.3, fill_color='#538d22', level='underlay', line_color=None) # green
    ba3 = BoxAnnotation(bottom=7, top=9, left=0, right=9, fill_alpha=1, fill_color='#fbe5d6', level='underlay', line_color=None) # orange
    ba2 = BoxAnnotation(bottom=0, top=7, left=7, right=9, fill_alpha=1, fill_color='#fbe5d6', level='underlay', line_color=None) # orange
    ba1 = BoxAnnotation(bottom=0, top=7, left=0, right=7, fill_alpha=0.3, fill_color='#bf0603', level='underlay',line_color=None) # red
    general_dict[points_plot_name].add_layout(ba1)
    general_dict[points_plot_name].add_layout(ba2)
    general_dict[points_plot_name].add_layout(ba3)
    general_dict[points_plot_name].add_layout(ba4)
    general_dict[points_plot_name].add_layout(ba5)

#----------------------------- ^ Static Background ^ -----------------------------#
#---------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------#
#-------------------------------- Utility Methods --------------------------------#

def calculate_points(type_filter):

    df_tempy = general_dict['individual_filtered_df'][type_filter].copy()

    if len(df_tempy) == 0:

        barycenter = np.array([-10.0, -10.0])
        bary_data = pd.DataFrame(columns=['recommendation', 'easiness'])

        general_dict['individual_filtered_df_bary'][type_filter] = bary_data
        general_dict['individual_filtered_df_bary_p'][type_filter] = barycenter
        general_dict['individual_filtered_df'][type_filter] = df_tempy
        return pd.DataFrame(columns=[])

    arr_slice = df_tempy[['recommendation', 'easiness']].values
    lidx = np.ravel_multi_index(arr_slice.T,arr_slice.max(0)+1)
    unq,unqtags,counts = np.unique(lidx,return_inverse=True,return_counts=True)
    df_tempy["visual_sum"] = counts[unqtags]

    # Create visual barycenter with edges
    barycenter = df_tempy[['recommendation', 'easiness']].astype({'recommendation':'float32', 'easiness':'float32'}).mean().to_numpy()

    # Create barycenter dataframe
    bary_numpy = df_tempy[['recommendation', 'easiness']].astype({'recommendation':'float32', 'easiness':'float32'}).to_numpy()

    row_bary = [barycenter[0], barycenter[1]]
    row_empty = np.empty((1,bary_numpy.shape[1]))
    row_empty.fill(np.nan)

    bary_numpy = np.insert(bary_numpy, range(1, len(bary_numpy)+1, 1), row_bary, axis=0)
    bary_numpy = np.insert(bary_numpy, range(2, len(bary_numpy), 2), row_empty, axis=0)
    bary_data = pd.DataFrame(bary_numpy, columns=['recommendation', 'easiness'])


    general_dict['individual_filtered_df'][type_filter] = df_tempy
    general_dict['individual_filtered_df_bary'][type_filter] = bary_data
    general_dict['individual_filtered_df_bary_p'][type_filter] = barycenter


#------------------------------ ^ Utility Methods ^ ------------------------------#
#---------------------------------------------------------------------------------#

# Create data table structure
data_columns = [
        # TableColumn(field="filter_one", title="Name"),
        # TableColumn(field="filter_two", title="Service"),
        # TableColumn(field="filter_three", title="Factory"),
        TableColumn(field="filter_four", title="Segment"),
        TableColumn(field="recommendation", title="Recommendation"),
        TableColumn(field="easiness", title="Easiness"),
    ]
# data_source = ColumnDataSource(pd.DataFrame(columns=['filter_one', 'filter_two', 'filter_three', 'filter_four', 'recommendation', 'easiness']))
data_source = ColumnDataSource(pd.DataFrame(columns=['filter_four', 'recommendation', 'easiness']))
data_table = DataTable(source=data_source, columns=data_columns, width=400, height=200, sizing_mode='stretch_width')

def typed_callback_c(type_filter):
    cdf_points = general_dict[type_filter]

    recommendations, easinesses = ([],[])

    inds = cdf_points.selected.indices
    if (len(inds) == 0):
        pass

    for i in range(0, len(inds)):
        recommendations.append(cdf_points.data['recommendation'][inds[i]])
        easinesses.append(cdf_points.data['easiness'][inds[i]])
    
    df_points = general_dict['individual_filtered_df'][type_filter]
    # current = df_points.loc[(df_points['recommendation'].isin(recommendations)) & (df_points['easiness'].isin(easinesses)) & (df_points['filter_one'].isin(general_dict['individual_filters'][type_filter])) & (df_points['filter_two'].isin(general_dict['individual_filters'][type_filter])) & (df_points['filter_three'].isin(general_dict['individual_filters'][type_filter])) & (df_points['filter_four'].isin(general_dict['individual_filters'][type_filter]))]
    current = df_points.loc[(df_points['recommendation'].isin(recommendations)) & (df_points['easiness'].isin(easinesses)) & (df_points['filter_four'].isin(general_dict['individual_filters'][type_filter]))]
    
    data_source.data = {
            # 'filter_one' : current.filter_one,
            # 'filter_two' : current.filter_two,
            # 'filter_three' : current.filter_three,
            'filter_four' : current.filter_four,
            'recommendation' : current.recommendation,
            'easiness' : current.easiness,
        }

# Update data table on circle tap
def callback_c0(attr, old, new):
    c_type_filter = general_dict['type_list'][0]
    typed_callback_c(c_type_filter)

def callback_c1(attr, old, new):
    c_type_filter = general_dict['type_list'][1]
    typed_callback_c(c_type_filter)

def callback_c2(attr, old, new):
    c_type_filter = general_dict['type_list'][2]
    typed_callback_c(c_type_filter)


def init_plot_points(type_filter):
    # Calculate filtered dataframe
    calculate_points(type_filter)
    # Create ColumnDataSource
    general_dict[type_filter] = ColumnDataSource(general_dict['individual_filtered_df'][type_filter])

    plot_name = type_filter+'_plot'
    # Create figure
    create_points_plot_layout(plot_name)
    # Plot circles
    general_dict[plot_name].circle('recommendation', 'easiness', name=type_filter, size='visual_sum', source=general_dict[type_filter], selection_fill_alpha=0.2, selection_color="firebrick", line_width=1, nonselection_line_color="firebrick")

    # Attach circle tap callback to circles
    if type_filter == general_dict['type_list'][0]:
        general_dict[type_filter].selected.on_change('indices', callback_c0)
    elif type_filter == general_dict['type_list'][1]:
        general_dict[type_filter].selected.on_change('indices', callback_c1)
    elif type_filter == general_dict['type_list'][2]:
        general_dict[type_filter].selected.on_change('indices', callback_c2)
    
    # Plot barycenter and connecting edges
    general_dict['bary_points'][type_filter] = ColumnDataSource(general_dict['individual_filtered_df_bary'][type_filter])
    general_dict[plot_name].line(x='recommendation', y='easiness', source=general_dict['bary_points'][type_filter], name='bary', line_width=1, level='overlay', color='#2a679d')
    general_dict['bary_p'][type_filter] = ColumnDataSource(dict(bary_x=[general_dict['individual_filtered_df_bary_p'][type_filter][0]], bary_y=[general_dict['individual_filtered_df_bary_p'][type_filter][1]]))
    general_dict[plot_name].circle(x='bary_x', y='bary_y', color='firebrick', size=20, name='barypoint', level='overlay', source=general_dict['bary_p'][type_filter])

    return general_dict[plot_name]

def update_plot_points(type_filter):
    # Calculate filtered dataframes
    calculate_points(type_filter)

    # Update ColumnDataSource
    general_dict[type_filter].data = general_dict['individual_filtered_df'][type_filter]
    general_dict['bary_points'][type_filter].data = general_dict['individual_filtered_df_bary'][type_filter]
    barycenter = general_dict['individual_filtered_df_bary_p'][type_filter]
    general_dict['bary_p'][type_filter].data = dict(bary_x=[general_dict['individual_filtered_df_bary_p'][type_filter][0]], bary_y=[general_dict['individual_filtered_df_bary_p'][type_filter][1]])

general_dict['points_plots'] = []
for type_filter in general_dict['type_list']:
    general_dict['points_plots'].append(init_plot_points(type_filter))

row_four = row(general_dict['points_plots'], sizing_mode='scale_width', css_classes=['super-flex'], name='row_four')

###################################################################################
###################################################################################


###################################################################################
############################ Visual 2 - Category Filters ##########################

# def callback_filter_3(attr, old, new):
#     update_filters()
#     filter_dfs()

#     for type_filter in general_dict['type_list']:
#         update_plot_points(type_filter)
#         update_emotions_points(type_filter)
#         update_frequency_points(type_filter)
#         update_wordcloud_points(type_filter)
#     # print("callback_filter_3", timeit.timeit('output = 10*5'))

# general_dict['filter_3'] = CheckboxButtonGroup(active=list(range(len(general_dict['full_factory_filter_list']))) , labels=general_dict['full_factory_filter_list'], sizing_mode='scale_width')#, name=row_one)

# general_dict['filter_3'].on_change('active', callback_filter_3)

# row_one = general_dict['filter_3']
#############################
# def callback_filter_1(attr, old, new):
#     update_filters()
#     filter_dfs()

#     for type_filter in general_dict['type_list']:
#         update_plot_points(type_filter)
#         update_emotions_points(type_filter)
#         update_frequency_points(type_filter)
#         update_wordcloud_points(type_filter)
#     # print("callback_filter_1", timeit.timeit('output = 10*5'))

# general_dict['filter_1'] = MultiSelect(value=general_dict['full_name_filter_list'], options=general_dict['full_name_filter_list'], height=130)
# button_1 = Button(label="Select All")
# button_2 = Button(label="Select None")

# button_1.js_on_event('button_click', CustomJS(args=dict(s=general_dict['filter_1']), code="s.value=s.options"))
# button_2.js_on_event('button_click', CustomJS(args=dict(s=general_dict['filter_1']), code="s.value=[]"))
# general_dict['filter_1'].on_change('value', callback_filter_1)


def callback_filter_4(attr, old, new):
    update_filters()
    filter_dfs()

    for type_filter in general_dict['type_list']:
        update_plot_points(type_filter)
        update_emotions_points(type_filter)
        update_frequency_points(type_filter)
        update_wordcloud_points(type_filter)
    # print("callback_filter_4", timeit.timeit('output = 10*5'))

general_dict['filter_4'] = MultiSelect(value=general_dict['full_segment_filter_list'], options=general_dict['full_segment_filter_list'], height=130)
button_3 = Button(label="Select All")
button_4 = Button(label="Select None")

button_3.js_on_event('button_click', CustomJS(args=dict(s=general_dict['filter_4']), code="s.value=s.options"))
button_4.js_on_event('button_click', CustomJS(args=dict(s=general_dict['filter_4']), code="s.value=[]"))
general_dict['filter_4'].on_change('value', callback_filter_4)


# def callback_filter_2(attr, old, new):
#     update_filters()
#     filter_dfs()

#     for type_filter in general_dict['type_list']:
#         update_plot_points(type_filter)
#         update_emotions_points(type_filter)
#         update_frequency_points(type_filter)
#         update_wordcloud_points(type_filter)
#     # print("callback_filter_2", timeit.timeit('output = 10*5'))

# general_dict['filter_2'] = MultiSelect(value=general_dict['full_service_filter_list'], options=general_dict['full_service_filter_list'], height=130)#, name=row_one)

# button_5 = Button(label="Select All")
# button_6 = Button(label="Select None")

# button_5.js_on_event('button_click', CustomJS(args=dict(s=general_dict['filter_2']), code="s.value=s.options"))
# button_6.js_on_event('button_click', CustomJS(args=dict(s=general_dict['filter_2']), code="s.value=[]"))
# general_dict['filter_2'].on_change('value', callback_filter_2)


# buttons_row = row(column(children=[general_dict['filter_1'], button_1, button_2], height=200, max_height=200), column(children=[general_dict['filter_4'], button_3, button_4], height=200, max_height=200, width=350), data_table)
buttons_row = row(column(children=[general_dict['filter_4'], button_3, button_4], height=200, max_height=200, width=320), data_table)



###################################################################################
###################################################################################

###################################################################################
############################ Visual 3 - Service Filters ###########################

# def callback_individual_filter_v0(attr, old, new):
#     update_filter(general_dict['type_list'][0])
#     filter_df(general_dict['type_list'][0])
#     update_plot_points(general_dict['type_list'][0])
#     update_emotions_points(general_dict['type_list'][0])
#     update_frequency_points(general_dict['type_list'][0])
#     update_wordcloud_points(general_dict['type_list'][0])
#     # print("callback_individual_filter_v0 ", timeit.timeit('output = 10*5'))

# def callback_individual_filter_v1(attr, old, new):
#     update_filter(general_dict['type_list'][1])
#     filter_df(general_dict['type_list'][1])
#     update_plot_points(general_dict['type_list'][1])
#     update_emotions_points(general_dict['type_list'][1])
#     update_frequency_points(general_dict['type_list'][1])
#     update_wordcloud_points(general_dict['type_list'][1])
#     # print("callback_individual_filter_v1 ", timeit.timeit('output = 10*5'))

# def callback_individual_filter_v2(attr, old, new):
#     update_filter(general_dict['type_list'][2])
#     filter_df(general_dict['type_list'][2])
#     update_plot_points(general_dict['type_list'][2])
#     update_emotions_points(general_dict['type_list'][2])
#     update_frequency_points(general_dict['type_list'][2])
#     update_wordcloud_points(general_dict['type_list'][2])
#     # print("callback_individual_filter_v2 ", timeit.timeit('output = 10*5'))


# general_dict['individual_filters_vis'] = {}
# for type_filter in general_dict['type_list']:
#     options = general_dict['individual_service_filter_list'][type_filter]
#     general_dict['individual_filters_vis'][type_filter] = MultiChoice(value=options, options=options, sizing_mode='scale_both', name='individual_filter_'+type_filter)
#     if type_filter == general_dict['type_list'][0]:
#         general_dict['individual_filters_vis'][type_filter].on_change('value', callback_individual_filter_v0)
#     elif type_filter == general_dict['type_list'][1]:
#         general_dict['individual_filters_vis'][type_filter].on_change('value', callback_individual_filter_v1)
#     elif type_filter == general_dict['type_list'][2]:
#         general_dict['individual_filters_vis'][type_filter].on_change('value', callback_individual_filter_v2)

# row_three = row(general_dict['individual_filters_vis'][general_dict['type_list'][0]],general_dict['individual_filters_vis'][general_dict['type_list'][1]],general_dict['individual_filters_vis'][general_dict['type_list'][2]], sizing_mode='scale_width', css_classes=['super-flex'], name='row_three')

###################################################################################
###################################################################################

###################################################################################
############################# Visual 4 - Emotions Plot ############################

def create_emotions_plot_layout(points_plot_name):

    general_dict[points_plot_name] = Plot(
        title=None, plot_width=600, plot_height=180,
        min_border=0, toolbar_location=None, outline_line_color=None, output_backend="webgl")

    general_dict[points_plot_name].add_glyph(HBar(y=0.4, right=0, left=-100, height=0.2, fill_color="#931a25", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.0, right=0, left=-100, height=0.2, fill_color="#931a25", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.4, right=30, left=0, height=0.2, fill_color="#ffc93c", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.0, right=30, left=0, height=0.2, fill_color="#ffc93c", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.4, right=70, left=30, height=0.2, fill_color="#b3de69", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.0, right=70, left=30, height=0.2, fill_color="#b3de69", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.4, right=100, left=70, height=0.2, fill_color="#158467", line_width=0))
    general_dict[points_plot_name].add_glyph(HBar(y=0.0, right=100, left=70, height=0.2, fill_color="#158467", line_width=0))

    # Create labels
    citation = Label(y=0.55, text='Recommendation', text_align='center', render_mode='css', text_color="#4c4c4c", text_font_style='bold')
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(y=0.16, text='Easiness', text_align='center', render_mode='css', text_color="#4c4c4c", text_font_style='bold')
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(x=-86, y=-0.2, text='NEEDS IMPROVEMENT', text_font_size='1vw', render_mode='css', text_color="#931a25")
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(x=7, y=-0.2, text='GOOD', text_font_size='1vw', render_mode='css', text_color="#ffc93c")
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(x=40, y=-0.2, text='GREAT', text_font_size='1vw', render_mode='css', text_color="#b3de69")
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(x=68, y=-0.2, text='EXCELLENT', text_font_size='1vw', render_mode='css', text_color="#158467")
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(x=-103, y=0.23, text='-100', text_font_size='1vw', render_mode='css', text_color="#4c4c4c")
    general_dict[points_plot_name].add_layout(citation)
    citation = Label(x=93, y=0.23, text='100', text_font_size='1vw', render_mode='css', text_color="#4c4c4c")
    general_dict[points_plot_name].add_layout(citation)



    # citation = Label(x=1.5, y=0.35, text='0', render_mode='css', text_color="#f4f4f4")
    # general_dict[points_plot_name].add_layout(citation)
    # citation = Label(x=31.5, y=0.35, text='30', render_mode='css', text_color="#f4f4f4")
    # general_dict[points_plot_name].add_layout(citation)
    # citation = Label(x=71.5, y=0.35, text='70', render_mode='css', text_color="#f4f4f4")
    # general_dict[points_plot_name].add_layout(citation)
    # citation = Label(x=1.5, y=-0.05, text='0', render_mode='css', text_color="#f4f4f4")
    # general_dict[points_plot_name].add_layout(citation)
    # citation = Label(x=31.5, y=-0.05, text='30', render_mode='css', text_color="#f4f4f4")
    # general_dict[points_plot_name].add_layout(citation)
    # citation = Label(x=71.5, y=-0.05, text='70', render_mode='css', text_color="#f4f4f4")
    # general_dict[points_plot_name].add_layout(citation)


def init_emotions_points(type_filter):

    df_emotions = general_dict['individual_df'][type_filter][['rec_sc','eas_sc']]

    rec_score = round(df_emotions['rec_sc'].mean() * 100, 2)
    easy_score = round(df_emotions['eas_sc'].mean() * 100, 2)

    plot_name = type_filter+'_emotions_plot'

    create_emotions_plot_layout(plot_name)

    if 'emotions_rec_score' not in general_dict:
        general_dict['emotions_rec_score'] = {}
    if 'emotions_easy_score' not in general_dict:
        general_dict['emotions_easy_score'] = {}

    general_dict['emotions_rec_score'][type_filter] = ColumnDataSource(dict(right=[rec_score], left=[rec_score], text=[rec_score], text_x=[rec_score+2.5]))
    general_dict['emotions_easy_score'][type_filter] = ColumnDataSource(dict(right=[easy_score], left=[easy_score], text=[easy_score], text_x=[easy_score+2.5]))

    citation = LabelSet(x='text_x', y=0.43, text='text', text_font_size='1vw', render_mode='css', text_color="black", source=general_dict['emotions_rec_score'][type_filter])
    general_dict[plot_name].add_layout(citation)
    citation = LabelSet(x='text_x', y=0.03, text='text', text_font_size='1vw', render_mode='css', text_color="black", source=general_dict['emotions_easy_score'][type_filter])
    general_dict[plot_name].add_layout(citation)

    general_dict[plot_name].add_glyph(general_dict['emotions_rec_score'][type_filter], HBar(y=0.4, right='right', left='left', height=0.2, fill_color="#1a1c20", line_width=4), name='rec_s')
    general_dict[plot_name].add_glyph(general_dict['emotions_easy_score'][type_filter], HBar(y=0.0, right='right', left='left', height=0.2, fill_color="#1a1c20", line_width=4), name='easy_s')

    return general_dict[plot_name]


def update_emotions_points(type_filter):
    # Calculate filtered dataframe
    df_emotions = general_dict['individual_filtered_df'][type_filter][['rec_sc','eas_sc']]

    rec_score = round(df_emotions['rec_sc'].mean() * 100, 2)
    easy_score = round(df_emotions['eas_sc'].mean() * 100, 2)

    if np.isnan(rec_score) or np.isnan(easy_score):
        rec_score = 0.0
        easy_score = 0.0

    # Update ColumnDataSource
    general_dict['emotions_rec_score'][type_filter].data = dict(right=[rec_score], left=[rec_score], text=[rec_score], text_x=[rec_score+2.5])
    general_dict['emotions_easy_score'][type_filter].data = dict(right=[easy_score], left=[easy_score], text=[easy_score], text_x=[easy_score+2.5])

####################################
general_dict['emotions_plots'] = []
for type_filter in general_dict['type_list']:
    general_dict['emotions_plots'].append(init_emotions_points(type_filter))

row_five = row(general_dict['emotions_plots'], sizing_mode='scale_width', css_classes=['super-flex'], name='row_five', margin=(30,0,0,0))

###################################################################################
###################################################################################

###################################################################################
############################# Visual 5 - Frequency Plot ###########################

def create_frequency_plot_layout(points_plot_name, y_range):

    general_dict[points_plot_name] = figure(y_range=y_range, plot_height=550, plot_width=500, match_aspect=True, toolbar_location=None, tools="", outline_line_color=None, output_backend="webgl", name='freq_f')

    general_dict[points_plot_name].title.text = "Frequencies"
    general_dict[points_plot_name].title.align = "center"
    general_dict[points_plot_name].title.text_color = "#4c4c4c"
    general_dict[points_plot_name].title.render_mode = 'css'
    general_dict[points_plot_name].title.text_font_size = "16px"
    general_dict[points_plot_name].title.text_font_style='bold'

    general_dict[points_plot_name].xgrid.grid_line_color = None
    general_dict[points_plot_name].ygrid.grid_line_color = None
    general_dict[points_plot_name].x_range.start = 0.0

def init_frequency_points(type_filter):
    # To change!!!
    # filter_columns = ['filter_one', 'filter_two', 'filter_three', 'filter_four']
    filter_columns = ['filter_four']

    filtered_df = general_dict['individual_df'][type_filter]
    df_one = filtered_df.loc[(~filtered_df['question_one_filtered_lemmas'].isna()), ['sentiment', 'question_one_filtered_lemmas'] + filter_columns]

    d_pv = dict()

    selected_lemmas_pv = df_one.loc[(df_one['sentiment'] > 0.0) & (df_one['question_one_filtered_lemmas'].str.len() > 1)][ ['question_one_filtered_lemmas'] + filter_columns]
    for lemmas_row in selected_lemmas_pv.itertuples():
        for word in lemmas_row[1]:
            row_list = list(lemmas_row[2:])
            if not word in d_pv:
                d_pv[word] = np.array([row_list])
            d_pv[word] = np.vstack((d_pv[word],np.array(  [row_list]  ) ))

    d_uv = dict()

    selected_lemmas_pv = df_one.loc[(df_one['sentiment'] == 0.0) & (df_one['question_one_filtered_lemmas'].str.len() > 1)][['question_one_filtered_lemmas'] + filter_columns]
    for lemmas_row in selected_lemmas_pv.itertuples():
        for word in lemmas_row[1]:
            row_list = list(lemmas_row[2:])
            if not word in d_uv:
                d_uv[word] = np.array([row_list])
            d_uv[word] = np.vstack((d_uv[word],np.array(  [row_list]  ) ))

    d_nv = dict()

    selected_lemmas_pv = df_one.loc[(df_one['sentiment'] < 0.0) & (df_one['question_one_filtered_lemmas'].str.len() > 1)][['question_one_filtered_lemmas'] + filter_columns]
    for lemmas_row in selected_lemmas_pv.itertuples():
        for word in lemmas_row[1]:
            row_list = list(lemmas_row[2:])
            if not word in d_nv:
                d_nv[word] = np.array([row_list])
            d_nv[word] = np.vstack((d_nv[word],np.array(  [row_list]  ) ))
    
    def if_cond_list(item):
        for subitem in item:
            if subitem in general_dict['individual_filters'][type_filter]:
                pass
            else:
                return False
        return True

    d_freq_pv = {k: sum([1 if if_cond_list(item) else 0 for item in v[:]]) for k, v in d_pv.items()}
    d_freq_uv = {k: sum([1 if if_cond_list(item) else 0 for item in v[:]]) for k, v in d_uv.items()}
    d_freq_uv = pd.DataFrame.from_dict({k: d_freq_uv[k] for k in d_freq_pv.keys() if k in d_freq_uv}, orient='index', columns=['freq_uv']).reset_index()
    d_freq_nv = {k: sum([1 if if_cond_list(item) else 0 for item in v[:]]) for k, v in d_nv.items()}
    d_freq_nv = pd.DataFrame.from_dict({k: d_freq_nv[k] for k in d_freq_pv.keys() if k in d_freq_nv}, orient='index', columns=['freq_nv']).reset_index()
    d_freq_pv = pd.DataFrame.from_dict(d_freq_pv, orient='index', columns=['freq_pv']).reset_index()
    frequency_df = d_freq_pv.merge(d_freq_uv, how='left').merge(d_freq_nv, how='left').fillna(0).astype({'index':'category', 'freq_pv':'int16', 'freq_uv':'int16', 'freq_nv':'int16'}).rename(columns = {'index':'freq_w'})


    general_dict['freq_df'][type_filter] = frequency_df
    general_dict['freq_words_slice'][type_filter] = slice(len(frequency_df))
    general_dict['freq_source'][type_filter] = ColumnDataSource(frequency_df)
    general_dict['d_pv'][type_filter] = d_pv
    general_dict['d_uv'][type_filter] = d_uv
    general_dict['d_nv'][type_filter] = d_nv

    plot_name = type_filter+'_frequency_plot'

    create_frequency_plot_layout(plot_name, frequency_df['freq_w'].to_list())

    general_dict[plot_name].hbar_stack(['freq_pv', 'freq_uv', 'freq_nv'], y='freq_w', width=0.9, color=['#29ce42', '#ffc12f', '#ff473d'], source=general_dict['freq_source'][type_filter], name='freq_w')

    return general_dict[plot_name]

def update_frequency_points(type_filter):
    d_pv = general_dict['d_pv'][type_filter]
    d_uv = general_dict['d_uv'][type_filter]
    d_nv = general_dict['d_nv'][type_filter]

    def if_cond_list(item):
        for subitem in item:
            if subitem in general_dict['individual_filters'][type_filter]:
                pass
            else:
                return False
        return True

    # Calculate new word frequencies
    d_freq_pv = {k: sum([1 if if_cond_list(item) else 0 for item in v[:]]) for k, v in d_pv.items()}
    general_dict['dict_freq_pv'][type_filter] = d_freq_pv
    d_freq_uv = {k: sum([1 if if_cond_list(item) else 0 for item in v[:]]) for k, v in d_uv.items()}
    d_freq_uv = pd.DataFrame.from_dict({k: d_freq_uv[k] for k in d_freq_pv.keys() if k in d_freq_uv}, orient='index', columns=['freq_uv']).reset_index()
    d_freq_nv = {k: sum([1 if if_cond_list(item) else 0 for item in v[:]]) for k, v in d_nv.items()}
    d_freq_nv = pd.DataFrame.from_dict({k: d_freq_nv[k] for k in d_freq_pv.keys() if k in d_freq_nv}, orient='index', columns=['freq_nv']).reset_index()
    d_freq_pv = pd.DataFrame.from_dict(d_freq_pv, orient='index', columns=['freq_pv']).reset_index()
    frequency_df = d_freq_pv.merge(d_freq_uv, how='left').merge(d_freq_nv, how='left').fillna(0).astype({'index':'category', 'freq_pv':'int16', 'freq_uv':'int16', 'freq_nv':'int16'}).rename(columns = {'index':'freq_w'})
    general_dict['d_freq_pv'][type_filter] = d_freq_pv
    
    # Update word frequencies
    general_dict['freq_source'][type_filter].patch({ 'freq_pv' : [(general_dict['freq_words_slice'][type_filter],frequency_df['freq_pv'])], 'freq_uv' : [(general_dict['freq_words_slice'][type_filter],frequency_df['freq_uv'])], 'freq_nv' : [(general_dict['freq_words_slice'][type_filter],frequency_df['freq_nv'])] })

general_dict['frequencies_plots'] = []
for type_filter in general_dict['type_list']:
    general_dict['frequencies_plots'].append(init_frequency_points(type_filter))

row_six = row(general_dict['frequencies_plots'], sizing_mode='scale_width', css_classes=['super-flex'], name='row_six')

###################################################################################
###################################################################################

###################################################################################
############################### Visual 6 - Words Cloud ############################

from wordcloud import WordCloud

def create_wordcloud_plot_layout(points_plot_name):

    general_dict[points_plot_name] = figure(width = 400, height=200, toolbar_location=None, tools="", output_backend="webgl")
    general_dict[points_plot_name].title.text = "Word Cloud"
    general_dict[points_plot_name].title.align = "center"
    general_dict[points_plot_name].title.text_color = "#4c4c4c"
    general_dict[points_plot_name].title.render_mode = 'css'
    general_dict[points_plot_name].title.text_font_size = "16px"
    general_dict[points_plot_name].title.text_font_style='bold'

    # Hide axis, grid, padding
    general_dict[points_plot_name].axis.visible = False
    general_dict[points_plot_name].xgrid.grid_line_color = None
    general_dict[points_plot_name].ygrid.grid_line_color = None
    general_dict[points_plot_name].x_range.range_padding = 0
    general_dict[points_plot_name].y_range.range_padding = 0


def init_wordcloud_points(type_filter):
    
    plot_name = type_filter+'_wordcloud_plot'

    create_wordcloud_plot_layout(plot_name)

    # Prepare dataframe
    frequency_df = general_dict['freq_df'][type_filter]

    df_viso = frequency_df[['freq_w','freq_pv']].set_index('freq_w')

    if df_viso.to_dict()['freq_pv']:
        # Calculate frequency-based word cloud
        general_dict['wordcloud'][type_filter] = WordCloud(background_color ='white')
        general_dict['wordcloud'][type_filter].generate_from_frequencies(frequencies=df_viso.to_dict()['freq_pv'])
        general_dict[plot_name].image(image=[np.flipud(np.mean(general_dict['wordcloud'][type_filter].to_array(), axis=2))], x=0, y=0, dw=200, dh=400, level="image", name='words')
    
    return general_dict[plot_name]

def update_wordcloud_points(type_filter):
    
    plot_name = type_filter+'_wordcloud_plot'

    # Remove old word cloud
    if len(general_dict[plot_name].select(name='words')) > 0:
        general_dict[plot_name].renderers.remove(general_dict[plot_name].select(name='words')[0])

    # Plot new word cloud
    if general_dict['d_freq_pv'][type_filter]['freq_pv'].sum() != 0:
        # Calculate frequency-based word cloud
        general_dict['wordcloud'][type_filter].generate_from_frequencies(frequencies=general_dict['dict_freq_pv'][type_filter])

        general_dict[plot_name].image(image=[np.flipud(np.mean(general_dict['wordcloud'][type_filter].to_array(), axis=2))], x=0, y=0, dw=200, dh=400, level="image", name='words')

general_dict['wordcloud_plots'] = []
for type_filter in general_dict['type_list']:
    general_dict['wordcloud_plots'].append(init_wordcloud_points(type_filter))

row_seven = row(general_dict['wordcloud_plots'], sizing_mode='scale_width', css_classes=['super-flex'], name='row_seven')


###################################################################################
###################################################################################

# curdoc().add_root(column(column(row_one,buttons_row, row_three,row_four,row_five,Spacer(height=50),row_six,row_seven, sizing_mode='scale_width', css_classes=['upper-flex']), sizing_mode='scale_width', margin=(0,30,0,0)))
curdoc().add_root(column(column(buttons_row,row_four,row_five,Spacer(height=50), row_six,row_seven, sizing_mode='scale_width', css_classes=['upper-flex']), sizing_mode='scale_width', margin=(0,30,0,0)))

