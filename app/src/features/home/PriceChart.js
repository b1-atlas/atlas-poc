import React from 'react';
import { Chart, Tooltip, Geom, Legend, Axis } from 'bizcharts';
import DataSet from '@antv/data-set';
import Slider from 'bizcharts-plugin-slider';
import autoHeight from './autoHeight';

// @autoHeight()
export default class PriceChart extends React.Component {
  render() {
    const {
      title,
      height = 500,
      padding = [60, 20, 40, 40],
      titleMap = {
        value: 'value',
      },
      borderWidth = 2,
      data = [
        {
          date: 0,
          value: 0,
        },
      ],
    } = this.props;

    data.sort((a, b) => a.date - b.date);

    let max;
    if (data[0] && data[0].value) {
      max = Math.max([...data].sort((a, b) => b.value - a.value)[0].value);
    }

    const ds = new DataSet({
      state: {
        start: data[0].date,
        end: data[data.length - 1].date,
      },
    });

    const dv = ds.createView();
    dv.source(data)
      .transform({
        type: 'filter',
        callback: obj => {
          const date = obj.date;
          return date <= ds.state.end && date >= ds.state.start;
        },
      })
      .transform({
        type: 'map',
        callback(row) {
          const newRow = { ...row };
          newRow[titleMap.value] = row.value;
          return newRow;
        },
      })
      .transform({
        type: 'fold',
        fields: [titleMap.value], // 展开字段集
        key: 'key', // key字段
        value: 'value', // value字段
      });

    console.log('data', data);
    console.log('dv', dv);

    const timeScale = {
      type: 'time',
      tickInterval: 1,
      mask: 'YYYY-MM-DD',
      range: [0, 1],
    };

    const cols = {
      date: timeScale,
      value: {
        max,
        min: 0,
      },
    };

    const SliderGen = () => (
      <Slider
        padding={[0, padding[1] + 20, 0, padding[3]]}
        width="auto"
        height={26}
        xAxis="date"
        yAxis="value"
        // scales={{ date: timeScale }}
        data={data}
        start={ds.state.start}
        end={ds.state.end}
        backgroundChart={{ type: 'line' }}
        onChange={({ startValue, endValue }) => {
          ds.setState('start', startValue);
          ds.setState('end', endValue);
        }}
      />
    );
    // scale={cols}
    return (
      <div className="price-chart" style={{ height: height + 30 }}>
        <div>
          {title && <h4>{title}</h4>}
          <Chart height={height} padding={padding} data={dv} forceFit>
            <Axis name="x" />
            <Tooltip />
            <Legend name="key" position="top" />
            <Geom type="line" position="x*value" size={borderWidth} color="key" />
          </Chart>
          <div style={{ marginRight: -20 }}>
            <SliderGen />
          </div>
        </div>
      </div>
    );
  }
}
