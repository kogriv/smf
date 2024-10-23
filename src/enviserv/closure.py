from . import dictan
from .mylog import MyLogger

lg = MyLogger('scopylogger','INFO')
ll = 30

da = dictan.DictAnalyzer(lg,ll)

globals_dict_core, \
locals_dict_core = dict(globals()), dict(locals())
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,"-Core (after import/creating serv objects)---------")
lg.mylev(ll,"-Global scope- dict(globals()):--------------------")
da.print_dict(globals_dict_core,0,False,False)
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,"-Core (after import/creating serv objects)---------")
lg.mylev(ll,"-Local scope- dict(locals()):----------------------")
da.print_dict(locals_dict_core,0,False,False)
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,"-Defining functions--------------------------------")

def outer_func(name,old,weight):
    lg.mylev(ll,"---------------------------------------------------")
    lg.mylev(ll,"--inside outer func--------------------------------")
    nested_var_in_outer_func = 'nested_var_in_OUTER_func_value'
    lg.mylev(ll,"outer_func for: " + str(name) + ", " +\
             str(old) + ", weight: "+ str(weight))
    globals_dict_before_closure, \
    locals_dict_before_closure = dict(globals()), dict(locals())
    lg.mylev(ll,"---------------------------------------------------")
    lg.mylev(ll,"------globals-in-outer-before-closure--------------")
    da.print_dict(locals_dict_before_closure)
    lg.mylev(ll,"---------------------------------------------------")
    lg.mylev(ll,"------locals-in-outer-before-closure---------------")
    da.print_dict(locals_dict_before_closure)
    lg.mylev(ll,"--comparing-globals-core-vs-globals-in-outer-func--")
    da.compare_dicts_info(
                        da.dict_info(globals_dict_core),
                        da.dict_info(globals_dict_before_closure),
                        False,'direct',True
                        )

    lg.mylev(ll,"---------------------------------------------------")
    lg.mylev(ll,"--comparing-locals-core-vs-locals-in-outer-func---")
    da.compare_dicts_info(
                        da.dict_info(locals_dict_core),
                        da.dict_info(locals_dict_before_closure),
                        False,'direct',True
                        )


    def nested_func():
        lg.mylev(ll,"---------------------------------------------------")
        lg.mylev(ll,"--inside nested func-------------------------------")
        lg.mylev(ll,"--using only part of outer_func params-------------")
        nested_var_in_nested_func = 'nested_var_in_NESTED_func_value'
        lg.mylev(ll,"name: " + name + ", old: " + old)
        nested_var_in_outer_func_modified_in_nested = \
            nested_var_in_outer_func + "_modified"
        nested_dict_in_nested_func = da.dict_info({'some_key':nested_var_in_outer_func_modified_in_nested})

        globals_dict_in_closure, \
        locals_dict_in_closure = dict(globals()), dict(locals())

        lg.mylev(ll,"---------------------------------------------------")
        lg.mylev(ll,"-cmprng-globals-in-outer-vs-globals-in-nested-func-")
        da.compare_dicts_info(da.dict_info(globals_dict_before_closure),
                              da.dict_info(globals_dict_in_closure),
                              False,'direct',True)
        
        lg.mylev(ll,"---------------------------------------------------")
        lg.mylev(ll,"-comparing-locals-outer-vs-locals-in-nested-func---")
        da.compare_dicts_info(da.dict_info(locals_dict_before_closure),
                              da.dict_info(locals_dict_in_closure),
                              False,'direct',True)


    return nested_func

def check_closured(func):
    if '__closure__' in dir(func):
        return True
    return False

def get_func_code_dict(func, asmbl_len=20, verbose = False):
    code_dict = {}
    if hasattr(func, '__code__'):
        for i in dir(func.__code__):
            value = getattr(func.__code__, i)
            if i in {'_co_code_adaptive', 'co_code',
                     'co_linetable', 'co_lnotab'}:
                if asmbl_len > 0:
                    value = str(value)[:asmbl_len]+"...."
            code_dict[i] = value
            if verbose:
                lg.mylev(ll,f"{i}"+': '+str(value))
    else: code_dict['empty_code_for'] = func
    return code_dict


def get_func_closure_val_dict(func, asmbl_len=20, verbose=False):
    closure_dict = {}
    if hasattr(func, '__closure__'):
        for closure_cell in func.__closure__:
            cell_address = hex(id(closure_cell))  # Получаем адрес ячейки и преобразуем его в строку
            value = closure_cell.cell_contents
            if asmbl_len > 0:
                str_value = str(value)[:asmbl_len]
                if len(str_value) == asmbl_len:
                    value = str_value + "...."
            closure_dict[cell_address] = value
            if verbose:
                print(f"{cell_address}: {value}")
    else:
        closure_dict['empty_closure_for'] = func
    return closure_dict


def get_closure_variables(func, verbose=False):
    closure_variables = {}
    if hasattr(func, '__closure__'):
        closure = func.__closure__
        if closure is not None:
            # Получение списка имен переменных из объекта функции
            var_names = func.__code__.co_freevars
            for name, cell in zip(var_names, closure):
                closure_variables[name] = cell.cell_contents
                if verbose:
                    lg.mylev(ll,f"{name}"+\
                             ': '+str(closure_variables[name]))
    return closure_variables

lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----f = outer_func("Ivan","30",80)-----------------')
f = outer_func("Ivan","30",80)

lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----f()--------------------------------------------')
f()

lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----f.__code__-------------------------------------')
lg.mylev(ll,f.__code__)
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----get_func_code_dict(f)--------------------------')
f_code_dict = get_func_code_dict(f,20)
da.print_dict(f_code_dict)
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----f.__closure__----------------------------------')
print(f.__closure__)
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----get_func_closure_val_dict(f)-------------------')
f_closure_val_dict = get_func_closure_val_dict(f,50)
da.print_dict(f_closure_val_dict)
lg.mylev(ll,"---------------------------------------------------")
lg.mylev(ll,'----get_closure_variables(f)-----------------------')
f_closure_name_dict = get_closure_variables(f)
da.print_dict(f_closure_name_dict)
lg.mylev(ll,"---------------------------------------------------")


lg.mylev(ll,"---------------------------------------------------")
