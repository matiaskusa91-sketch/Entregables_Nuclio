def calcular_total_ingresos(dataframe):
    """Calula el importe ventas totales creando una ueva columna de ventas (pxq)"""
    dataframe["ventas_en_€"] = dataframe["quantity"] * dataframe["price"]
    total_ingresos = dataframe["ventas_en_€"].sum()
    return total_ingresos
total_ingresos = calcular_total_ingresos(df_entregable)
print(f"El ingreso total de todas las ventas fue de '{total_ingresos}'")

def centro_con_mas_ventas(dataframe):
    """ Devuelve elentro comercial con mayor ventas, medido como número de transeacciones"""
    ventas_por_centro = dataframe.groupby('shopping_mall')['invoice_no'].count()
    centro_mas_ventas = ventas_por_centro.idxmax()
    return centro_mas_ventas
