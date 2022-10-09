> Created on Fri Oct  7 10:28:02 2022 @author: Richie Bao-caDesign设计(cadesign.cn)

# Note

## 代码审核反馈

### 代码审核中的疑问

1. 超链接地址加脚注形式，于各章文尾，格式示例：① OpenStreetMap(OSM)（https://www.openstreetmap.org）；
2. 电子版python库均为代码格式（链接格式）表述，示例：[plotly](https://plotly.com/)；
3. 移除注释掉的代码；
4. 代码风格。

    > 参考引用：[Google 开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/#section-2)，[Python PEP-8编码风格指南中文版](https://alvin.red/2017/10/07/python-pep-8/)

    * 风格是为了代码具有更好的可读性；
    * 风格同一模块和同一函数保持一致，同一项目保持一致，同风格指南或共识性规范尽量保持一致；
    * 缩进风格统一，示例：
    ```python    
    # 同开始分界符(左括号)对齐(Aligned with opening delimiter)
    foo = long_function_name(var_one, var_two,
                             var_three, var_four)

    # 续行多缩进一级以同其他代码区别
    def long_function_name(
            var_one, var_two, var_three,
            var_four):
        print(var_one)

    # 悬挂缩进需要多缩进一级(space hanging indent; nothing on first line)
    foo = long_function_name(
        var_one, var_two,
        var_three, var_four)
    # Aligned with opening delimiter in a dictionary
    foo = {
        long_dictionary_key: value1 +
                           value2,                            
        ...
       }
    ```
    * 行长度，示例
    ```python
    # Python会将圆括号, 中括号和花括号中的行隐式的连接起来
    Yes: foo_bar(self, width, height, color='black', design=None, x='foo',
                  emphasis=None, highlight=0)

        if (width == 0 and height == 0 and
            color == 'red' and emphasis == 'strong'):

        # 如果一个文本字符串在一行放不下, 可以使用圆括号来实现隐式行连接    
        x = ('This will build a very long long '
             'long long long long long long string')    

        # 在注释中，如果必要，将长的URL放在一行上
        # http://www.example.com/us/developer/documentation/api/content/v2.0/csv_file_name_extension_full_specification.html
    ```
    * 序列元素尾部逗号
    ```python
    Yes:   golomb3 = [0, 1, 3]
    Yes:   golomb4 = [
            0,
            1,
            4,
            6,
        ]

    No:    golomb4 = [
           0,
           1,
           4,
           6
       ]       
    ```  
    * 空行: 顶级定义之间空两行, 方法定义之间空一行

    顶级定义之间空两行, 比如函数或者类定义. 方法定义, 类定义与第一个方法之间, 都应该空一行. 函数或方法中, 某些地方要是你觉得合适, 就空一行.

    * 空格

    括号内不要有空格
    ```python
    Yes: spam(ham[1], {eggs: 2}, [])

    No:  spam( ham[ 1 ], { eggs: 2 }, [ ] )
    ```
    不要在逗号, 分号, 冒号前面加空格, 但应该在它们后面加(除了在行尾)
    ```python
    Yes: if x == 4:
            print(x, y)
        x, y = y, x

    No:  if x == 4 :
         print(x , y)
     x , y = y , x        
    ``` 
    参数列表, 索引或切片的左括号前不应加空格.
    ```python
    Yes: spam(1)

    no: spam (1)

    Yes: dict['key'] = list[index]

    No:  dict ['key'] = list [index]
    ```
    在二元操作符两边都加上一个空格, 比如赋值(=), 比较(==, <, >, !=, <>, <=, >=, in, not in, is, is not), 布尔(and, or, not). 至于算术操作符两边的空格该如何使用, 需要你自己好好判断. 不过两侧务必要保持一致. 
    
    > USDA待确定

    ```python
    Yes: x == 1

    No:  x<1
    ```
    当 `=` 用于指示关键字参数或默认参数值时, 不要在其两侧使用空格. 但若存在类型注释的时候,需要在 `=` 周围使用空格.
    ```python
    Yes: def complex(real, imag=0.0): return magic(r=real, i=imag)
    Yes: def complex(real, imag: float = 0.0): return Magic(r=real, i=imag)

    No:  def complex(real, imag = 0.0): return magic(r = real, i = imag)
    No:  def complex(real, imag: float=0.0): return Magic(r = real, i = imag)
    ```
    不要用空格来垂直对齐多行间的标记, 因为这会成为维护的负担(适用于:, #, =等):
    ```python
    Yes:
        foo = 1000  # comment
        long_name = 2  # comment that should not be aligned

        dictionary = {
            "foo": 1,
            "long_name": 2,
            }

    No:
        foo       = 1000  # comment
        long_name = 2     # comment that should not be aligned

        dictionary = {
            "foo"      : 1,
            "long_name": 2,
            }         
    ```

    * 注释

    函数部分注释按`spyder`提供的方法，例如：

    ```python
    def recursive_factorial(n):
        '''
        阶乘计算。

        Parameters
        ----------
        n : int
            阶乘计算的末尾值.

        Returns
        -------
        int
            阶乘计算结果.

        '''
        if n==1:
            return 1
        else:
            return n*recursive_factorial(n-1)
    ```

    （统一为spyder规范类）--可根据具体情况调整，但是需要保持部分一致，例如基础实验部分一致，专项研究部分一致。

    `#`之后空一格

    ```python
    Yes:
        # the next element is i+1

    No:
        #the next element is i+1
    ```

    无需注释模块中的所有函数:

        1. 公共的API需要注释
        2. 在代码的安全性，清晰性和灵活性上进行权衡是否注释
        3. 对于容易出现类型相关的错误的代码进行注释
        4. 难以理解的代码请进行注释
        5. 若代码中的类型已经稳定，可以进行注释. 对于一份成熟的代码，多数情况下，即使注释了所有的函数，也不会丧失太多的灵活性.

    * 字符串

    `%` 格式化字符前后空一格
    ```python
    Yes:
        x = '%s, %s!' % (imperative, expletive)

    Noe:
        x = '%s, %s!'%(imperative, expletive)
    ```

    避免在循环中用+和+=操作符来累加字符串. 由于字符串是不可变的, 这样做会创建不必要的临时对象, 并且导致二次方而不是线性的运行时间. 作为替代方案, 你可以将每个子串加入列表, 然后在循环结束后用 `.join` 连接列表. (也可以将每个子串写入一个 `cStringIO.StringIO` 缓存中.)
    ```python
    Yes: items = ['<table>']
        for last_name, first_name in employee_list:
            items.append('<tr><td>%s, %s</td></tr>' % (last_name, first_name))
        items.append('</table>')
        employee_table = ''.join(items)

    No: employee_table = '<table>'
        for last_name, first_name in employee_list:
            employee_table += '<tr><td>%s, %s</td></tr>' % (last_name, first_name)
        employee_table += '</table>'     
    ```
    在同一个文件中, 保持使用字符串引号的一致性. 使用单引号’或者双引号”之一用以引用字符串, 并在同一文件中沿用. 在字符串内可以使用另外一种引号, 以避免在字符串中使用.
    ```python
    Yes:
        Python('Why are you hiding your eyes?')
        Gollum("I'm scared of lint errors.")
        Narrator('"Good!" thought a happy Python reviewer.')

    No:
        Python("Why are you hiding your eyes?")
        Gollum('The lint. It burns. It burns us.')
        Gollum("Always the great lint. Watching. Watching.")     
    ```
    为多行字符串使用三重双引号`"""`而非三重单引号`'`. 当且仅当项目中使用单引号’来引用字符串时, 才可能会使用三重`'`为非文档字符串的多行字符串来标识引用. 文档字符串必须使用三重双引号`"`. 多行字符串不应随着代码其他部分缩进的调整而发生位置移动. 如果需要避免在字符串中嵌入额外的空间,可以使用串联的单行字符串或者使用 `textwrap.dedent()` 来删除每行多余的空间.
    ```python
    No:
    long_string = """This is pretty ugly.
    Don't do this.
    """

    Yes:
    long_string = """This is fine if your use case can accept
        extraneous leading spaces."""

    Yes:
    long_string = ("And this is fine if you cannot accept\n" +
        "extraneous leading spaces.")

    Yes:
    long_string = ("And this too is fine if you cannot accept\n"
        "extraneous leading spaces.")

    Yes:
    import textwrap

    long_string = textwrap.dedent("""\
        This is also fine, because textwrap.dedent()
        will collapse common leading spaces in each line.""")           
    ``` 

    *文件和sockets

    除文件外, sockets或其他类似文件的对象在没有必要的情况下打开, 会有许多副作用, 例如:

        1. 它们可能会消耗有限的系统资源, 如文件描述符. 如果这些资源在使用后没有及时归还系统, 那么用于处理这些对象的代码会将资源消耗殆尽.
        2. 持有文件将会阻止对于文件的其他诸如移动、删除之类的操作.
        3. 仅仅是从逻辑上关闭文件和sockets, 那么它们仍然可能会被其共享的程序在无意中进行读或者写操作. 只有当它们真正被关闭后, 对于它们尝试进行读或者写操作将会抛出异常, 并使得问题快速显现出来.

    而且, 幻想当文件对象析构时, 文件和sockets会自动关闭, 试图将文件对象的生命周期和文件的状态绑定在一起的想法, 都是不现实的. 因为有如下原因:

        1. 没有任何方法可以确保运行环境会真正的执行文件的析构. 不同的Python实现采用不同的内存管理技术, 比如延时垃圾处理机制. 延时垃圾处理机制可能会导致对象生命周期被任意无限制的延长.
        2. 对于文件意外的引用,会导致对于文件的持有时间超出预期(比如对于异常的跟踪, 包含有全局变量等).

    推荐使用 `with`语句 以管理文件:
    ```python
    with open("hello.txt") as hello_file:
        for line in hello_file:
            print line
    ```
    对于不支持使用”with”语句的类似文件的对象,使用 contextlib.closing():
    ```python
    import contextlib

    with contextlib.closing(urllib.urlopen("http://www.python.org/")) as front_page:
        for line in front_page:
            print line
    ```

    *导入格式 采用部分`Google 开源项目风格指南`

    导入应该按照从最通用到最不通用的顺序分组:

 	```python
    from __future__ import absolute_import # __future__ 导入
    import sys # 标准库导入
    import tensorflow as tf # 第三方库导入
    from otherproject.ai import mind # 本地代码子包导入
    ```   

    每种分组中, 应该根据每个模块的完整包路径按字典序排序, 忽略大小写.
    ```python
    import collections
    import queue
    import sys

    from absl import app
    from absl import flags
    import bs4
    import cryptography
    import tensorflow as tf

    from book.genres import scifi
    from myproject.backend import huxley
    from myproject.backend.hgwells import time_machine
    from myproject.backend.state_machine import main_loop
    from otherproject.ai import body
    from otherproject.ai import mind
    from otherproject.ai import soul

    # Older style code may have these imports down here instead:
    #from myproject.backend.hgwells import time_machine
    #from myproject.backend.state_machine import main_loop
    ```

    导入代码统一位置，1，位于模块文件开始行；2，位于函数内开始行。一次性在模块开始行调入，可以避免函数内重复调入。但如果导入的模块体量较大，且存在库之间的冲突，如果一次性在模块开始行导入，代码无法顺利运行，此时在函数内开始行调入较为合理。因此具体调入位置，根据具体情况比较确定。

    * 命名

    模块名写法: `module_name` ;包名写法: `package_name` ;类名: `ClassName` ;方法名: `method_name` ;异常名: `ExceptionName` ;函数名: `function_name` ;全局常量名: `GLOBAL_CONSTANT_NAME` ;全局变量名: `global_var_name` ;实例名: `instance_var_name` ;函数参数名: `function_parameter_name` ;局部变量名: `local_var_name` . 函数名,变量名和文件名应该是描述性的,尽量避免缩写,特别要避免使用非项目人员不清楚难以理解的缩写,不要通过删除单词中的字母来进行缩写. 始终使用 `.py`作为文件后缀名,不要用破折号.

    应该避免的名称:

        1. 单字符名称, 除了计数器和迭代器,作为 `try/except` 中异常声明的 `e`,作为 `with` 语句中文件句柄的 `f`.
        2. 包/模块名中的连字符(`-`)
        3. 双下划线开头并结尾的名称(Python保留, 例如`__init__`)

    命名约定:

        1. 所谓”内部(`Internal`)”表示仅模块内可用, 或者, 在类内是保护或私有的.
        2. 用单下划线(`_`)开头表示模块变量或函数是protected的(使用`from module import *`时不会包含).
        3. 用双下划线(`__`)开头的实例变量或方法表示类内私有.
        4. 将相关的类和顶级函数放在同一个模块里. 不像Java, 没必要限制一个类一个模块.
        5. 对类名使用大写字母开头的单词(如`CapWords`, 即`Pascal`风格), 但是模块名应该用小写加下划线的方式(如`lower_with_under.py`). 尽管已经有很多现存的模块使用类似于`CapWords.py`这样的命名, 但现在已经不鼓励这样做, 因为如果模块名碰巧和类名一致, 这会让人困扰.

    文件名:

        所有python脚本文件都应该以 `.py` 为后缀名且不包含 `-`.若是需要一个无后缀名的可执行文件,可以使用软联接或者包含 `exec "$0.py" "$@"` 的`bash`脚本. 

    Python之父Guido推荐的规范:

        | Type                       | Public             | Internal                                                          |
        |----------------------------|--------------------|-------------------------------------------------------------------|
        | Modules                    | lower_with_under   | _lower_with_under                                                 |
        | Packages                   | lower_with_under   |                                                                   |
        | Classes                    | CapWords           | _CapWords                                                         |
        | Exceptions                 | CapWords           |                                                                   |
        | Functions                  | lower_with_under() | _lower_with_under()                                               |
        | Global/Class Constants     | CAPS_WITH_UNDER    | _CAPS_WITH_UNDER                                                  |
        | Global/Class Variables     | lower_with_under   | _lower_with_under                                                 |
        | Instance Variables         | lower_with_under   | _lower_with_under (protected) or __lower_with_under (private)     |
        | Method Names               | lower_with_under() | _lower_with_under() (protected) or __lower_with_under() (private) |
        | Function/Method Parameters | lower_with_under   |                                                                   |
        | Local Variables            | lower_with_under   |                                                                   |

    * Main

    > 即使是一个打算被用作脚本的文件, 也应该是可导入的. 并且简单的导入不应该导致这个脚本的主功能(main functionality)被执行, 这是一种副作用. 主功能应该放在一个main()函数中.

    在Python中, `pydoc`以及单元测试要求模块必须是可导入的. 你的代码应该在执行主程序前总是检查 `if __name__ == '__main__'` , 这样当模块被导入时主程序就不会被执行.

    若使用 absl, 请使用 `app.run` :
    ```python
    from absl import app
    ...

    def main(argv):
        # process non-flag arguments
        ...

    if __name__ == '__main__':
        app.run(main)
    ```

    否则,使用:
    ```python
    def main():
        ...

    if __name__ == '__main__':
        main()
    ```

    所有的顶级代码在模块导入时都会被执行. 要小心不要去调用函数, 创建对象, 或者执行那些不应该在使用pydoc时执行的操作.

    * 类型注释（略）

    > 类型注释具体解释查看[Google 开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/#section-17)

5. 文件路径名需无空格，通常用`_`连接，例如： `xian_region_gdf=gpd.read_file('./data/xian_region/xian_region.shp')`；
6. 长代码换行（分行）；
7. `Json`数据格式化显示，[JSON Formatter](https://elmah.io/tools/json-formatter/) 或[在线JSON校验格式化工具](https://www.bejson.com/)
8. 参考文献：于各章末尾给出。格式参考期刊格式，固定一种。

### 代码建议 二 （截止到章节 2.1.3）

9. 公式格式标准化，用[TeX](https://atomurl.net/math/)；
10. 调整更新和敲定所有标题；
11. 规范术语（参考期刊要求），摘录如下（更新）：

    |  不规范 | 规范  |   |   |   |   |   |   |
    |---|---|---|---|---|---|---|---|
    | kstest  | K-S test  |   |   |   |   |   |   |
    | pvalue  | p-value  |   |   |   |   |   |   |
    | 变异数  |  Variance(方差) |   |   |   |   |   |   |
    | 共变异数  | Covariance(协方差)  |   |   |   |   |   |   |
    |   |   |   |   |   |   |   |   |
    |   |   |   |   |   |   |   |   |
    |   |   |   |   |   |   |   |   |1

12. 变量名，函数名等需再核实调整更新确认；
13. 文字语句核实调整更新确认；
14. 公式符号说明顺序同公式符号出现顺序；
15. 依赖库各章列表，给`requirements.txt`；


