import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';
import * as actions from './redux/actions';
import * as d3 from 'd3';
import moment from 'moment';
const dateformat = 'YYYY-MM-DD';

export class InteractiveChart extends Component {
  static propTypes = {
    home: PropTypes.object.isRequired,
    actions: PropTypes.object.isRequired,
    id: PropTypes.string.isRequired,
    text: PropTypes.string,
    measure: PropTypes.string.isRequired,
    unit: PropTypes.string,
    data: PropTypes.array.isRequired,
    forecastedData: PropTypes.array,
    startDate: PropTypes.string.isRequired,
    endDate: PropTypes.string.isRequired,
  };

  state = {
    adjustPoint: [],
  };

  componentWillMount() {
    // d3.select('#price-chart')
    //   .style('color', 'red')
    //   .html('Chart will mount');
  }

  componentDidMount() {
    const { data = [], forecastedData = [], startDate, endDate } = this.props;
    this.drawLineChart(data, forecastedData, startDate, endDate);
  }

  componentDidUpdate() {
    const { data = [], forecastedData = [], startDate, endDate } = this.props;
    this.drawLineChart(data, forecastedData, startDate, endDate);
  }

  componentWillReceiveProps(nextProps) {
    // if (nextProps.forecastedData.length > 0) {
    //   console.log(nextProps.forecastedData);
    //   const { data = [], startDate, endDate } = this.props;
    //   this.drawLineChart(data, nextProps.forecastedData, startDate, endDate);
    //   this.props.history.replace(`/position/view/${positionId}`);
    // }
  }

  drawLineChart = (data, forecastedData, startDate, endDate) => {
    const { id, yAxisText = 'Value', measure, unit = '' } = this.props;

    d3.select(`#${id}`)
      .selectAll('*')
      .remove();
    const start = moment(startDate, dateformat).toDate();
    const end = moment(endDate, dateformat).toDate();
    data = data.filter(d => {
      return d[measure] !== 0 && d['date'] >= start && d['date'] <= end;
    });
    // console.log('data', data);
    if (measure !== 'recommend') {
      forecastedData = forecastedData.filter(d => d[measure] !== 0);
    }
    let mergedData = [...data];
    if (forecastedData.length > 0) {
      if (data.length > 0) {
        forecastedData.unshift(data[data.length - 1]);
      }
      mergedData = [...mergedData, ...forecastedData];
    }
    const svgNode = this.node;
    let svgWidth = window.innerWidth - 160 < 800 ? 800 : window.innerWidth - 160;
    // if (data.length > 365) {
    //   svgWidth = data.length * 5;
    // }
    let svgHeight =
      parseInt(window.innerHeight - 500 / 2, 10) >= 500
        ? 500
        : parseInt(window.innerHeight - 500 / 2, 10);
    svgHeight = svgHeight < 400 ? 500 : svgHeight;
    // console.log(svgWidth);
    // console.log(svgHeight);
    const margin = { top: 20, right: 20, bottom: 80, left: 50 };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    d3.select(`div.tooltip.${measure}`).remove();
    const tooltip = d3
      .select(`.chart-container.${measure}`)
      .append('div')
      .attr('class', 'tooltip ' + measure)
      .style('opacity', 0);
    const dateTxt = tooltip.append('span').attr('class', 'date-txt');
    const valueTxt = tooltip.append('span').attr('class', 'val-txt');

    const svg = d3
      .select(svgNode)
      .attr('width', svgWidth)
      .attr('height', svgHeight);

    // const parseDate = d3.time.format('%d-%b-%y').parse;
    const bisectDate = d3.bisector(d => {
      return d['date'];
    }).left;
    const formatValue = d3.format(',.2f');
    const formatCurrency = d => {
      if (measure === 'quantity') {
        return d;
      }
      return unit + ' ' + formatValue(d);
    };

    const g = svg.append('g').attr('transform', `translate(${margin.left}, ${margin.top})`);

    const x = d3.scaleTime().rangeRound([0, width]);

    const y = d3.scaleLinear().rangeRound([height, 0]);

    const line = d3
      .line()
      .x(d => {
        return x(d['date']);
      })
      .y(d => {
        return y(d[measure]);
      });
    x.domain(
      d3.extent(mergedData, d => {
        return d['date'];
      }),
    );
    y.domain(
      d3.extent(mergedData, d => {
        return d[measure];
      }),
    );

    g.append('g')
      .attr('class', 'axis axis-x')
      .attr('transform', `translate(0, ${height})`)
      .call(customXAxis);

    function customXAxis(g) {
      const dataMonths =
        mergedData
          .map(d => d3.timeFormat('%Y-%m')(d['date']))
          .filter((v, i, a) => a.indexOf(v) === i) || [];
      const dateformat = dataMonths.length > 3 ? '%Y-%m' : '%Y-%m-%d';
      const xAxis = d3
        .axisBottom(x)
        .tickSizeInner(5) // the inner ticks will be of size 5
        .tickSizeOuter(0)
        .tickFormat(d3.timeFormat(dateformat)); //"%Y-%m-%d" %m-%d"
      g.call(xAxis);
      g.select('.domain')
        .attr('stroke', 'rgba(153, 153, 153, .6)')
        .attr('stroke-width', 1.5);
      // .tick:not(:first-of-type) line
      g.selectAll('.tick line')
        .attr('stroke', 'rgba(153, 153, 153, .6)')
        .attr('stroke-width', 1.5);
      g.selectAll('.tick text')
        .attr('fill', 'rgba(51, 51, 51, .85)')
        .attr('transform', 'translate(0, 20) rotate(45)');
    }

    g.append('g')
      .attr('class', 'axis axis-y')
      .call(customYAxis)
      .append('text')
      .attr('fill', 'rgb(102, 102, 102, .85)')
      .attr('transform', `translate(${measure === 'price' ? '22' : '30'}, -26)`)
      .attr('y', 7)
      .attr('dy', '0.71em')
      .attr('text-anchor', 'end')
      .style('font', '12px sans-serif')
      .text(yAxisText);

    g.append('path')
      .data([data])
      .attr('fill', 'none')
      .attr('stroke', 'rgba(24, 144, 255, .85)')
      .attr('stroke-linejoin', 'round')
      .attr('stroke-linecap', 'round')
      .attr('stroke-width', 2)
      .attr('d', line);

    if (forecastedData.length > 0) {
      g.append('path')
        .data([forecastedData])
        .attr('fill', 'none')
        .attr('stroke', 'rgb(255, 171, 3, .85)')
        .attr('stroke-linejoin', 'round')
        .attr('stroke-linecap', 'round')
        .attr('stroke-width', 2)
        .attr('d', line);
      const splitLine = svg
        .append('path')
        .attr('class', 'split-line')
        .attr('stroke', 'rgba(153, 153, 153, .45)')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 1.5);

      const lastDate = forecastedData[0]['date'];
      const splitX = x(lastDate) + margin.left;
      splitLine.attr('transform', `translate(0, 20)`).attr('d', `M${splitX},${height} ${splitX} 0`);
    }

    function customYAxis(g) {
      const yAxis = d3
        .axisRight(y)
        .tickSize(width)
        .tickFormat(function(d) {
          // var s = formatNumber(d / 1e6);
          return this.parentNode.nextSibling ? '\xa0' + d : unit + d;
        });
      g.call(yAxis);
      g.select('.domain').remove();
      // .tick:not(:first-of-type) line
      g.selectAll('.tick line')
        .attr('stroke', 'rgba(153, 153, 153, .45)')
        .attr('stroke-dasharray', '3 3')
        .attr('stroke-width', 1.5);
      g.selectAll('.tick text')
        .attr('fill', 'rgba(51, 51, 51, .85)')
        .attr('x', 4)
        .attr('dy', -4)
        .attr('transform', 'translate(-45, 6)');
    }

    const dot = svg
      .append('g')
      .attr('class', 'dot')
      .style('opacity', '0');

    const markLine = svg
      .append('path')
      .attr('class', 'mark-line')
      .attr('stroke', 'rgba(153, 153, 153, .85)')
      .attr('stroke-width', 1.5)
      .style('opacity', '0');

    dot
      .append('circle')
      .attr('r', 4.5)
      .style('cursor', 'pointer');

    svg
      .append('rect')
      .attr('class', 'overlay')
      .attr('width', width)
      .attr('height', height)
      .attr('transform', `translate(${margin.left}, ${margin.top})`)
      .on('mouseover', () => {
        dot.style('opacity', 1);
        tooltip.style('opacity', 0.9);
        markLine.style('opacity', 1);
      })
      .on('mouseout', () => {
        markLine.style('opacity', 0);
        tooltip.style('opacity', 0);
        dot.style('opacity', 0);
      })
      .on('mousemove', mousemove);

    function mousemove() {
      const x0 = x.invert(d3.mouse(this)[0]),
        i = bisectDate(mergedData, x0, 1),
        d0 = mergedData[i - 1],
        d1 = mergedData[i],
        d = x0 - d0['date'] > d1['date'] - x0 ? d1 : d0;
      const posX = x(d['date']) + margin.left;
      const posY = y(d[measure]) + margin.top;
      markLine.attr('transform', `translate(0, 20)`).attr('d', `M${posX},${height} ${posX} 0`);

      dot.attr('transform', `translate(${posX}, ${posY})`);
      const tooltipX = posX + 150 > svgWidth ? posX - 100 : posX + 35;
      const tooltipY = posY + 150 > svgHeight ? posY - 30 : posY + 35;
      tooltip.style('left', `${tooltipX}px`).style('top', `${tooltipY}px`);
      dateTxt.text(d3.timeFormat('%Y-%m-%d')(d['date']));
      valueTxt.text(formatCurrency(d[measure]));
    }
  };

  render() {
    return (
      <svg
        id={this.props.id}
        className={this.props.forecasting ? 'line-chart forecasting' : 'line-chart'}
        ref={node => (this.node = node)}
        width={'100%'}
        height={'100%'}
      />
    );
  }
}

/* istanbul ignore next */
function mapStateToProps(state) {
  return {
    home: state.home,
  };
}

/* istanbul ignore next */
function mapDispatchToProps(dispatch) {
  return {
    actions: bindActionCreators({ ...actions }, dispatch),
  };
}

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(InteractiveChart);
